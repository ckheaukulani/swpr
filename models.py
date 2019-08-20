import numpy as np
import tensorflow as tf

from gpflow import settings
from gpflow.decors import params_as_tensors
from gpflow.models.svgp import SVGP
import gpflow.training.monitor as gpmon


class DynamicCovarianceRegression(SVGP):
    """
    Effectively a helper wrapping call to SVGP.
    """
    def __init__(self, X, Y, kern, likelihood, Z, minibatch_size=None, whiten=True):
        """

        :param X: (N, 1)-array of inputs (time)
        :param Y: (N, D)-array of measurements
        :param kern: GPflow kernel object
        :param likelihood: GPflow likelihood object
        :param Z: (M, D)-array of initial
        :param minibatch_size:
        :param whiten:
        """
        cov_dim = likelihood.cov_dim
        nu = likelihood.nu

        # X and Y are the training dataset, subtract mean and save for prediction time
        Y_mean = np.mean(Y, axis=0)  # (D,)
        Y = Y - Y_mean  # (N, D)

        super().__init__(X, Y, kern, likelihood,
                         feat=None,
                         mean_function=None,
                         num_latent=cov_dim * nu,
                         q_diag=False,
                         whiten=whiten,
                         minibatch_size=minibatch_size,
                         Z=Z,
                         name='SVGP')  # must provide a name space when delaying build for GPflow opt functionality

        self.compile()  # if I don't compile now then there's a weird error when trying to construct the prediction
        self.Y_mean = Y_mean  # used when constructing the predictive densities
        self.construct_predictive_density()

    @params_as_tensors
    def construct_predictive_density(self):
        D = self.likelihood.D
        self.X_new = tf.placeholder(dtype=settings.float_type, shape=[None, 1])
        self.Y_new = tf.placeholder(dtype=settings.float_type, shape=[None, D])

        # subtract TRAINING mean
        Y_new = self.Y_new - self.Y_mean

        F, _ = self._build_predict(self.X_new)  # (N_new, num_latent); e.g., num_latent = D * nu
        N_new = tf.shape(F)[0]
        cov_dim = self.likelihood.cov_dim
        self.F_mu_new = tf.reshape(F, [N_new, cov_dim, -1])  # (N_new, cov_dim, nu)
        log_det_cov, yt_inv_y = self.likelihood.make_gaussian_components(self.F_mu_new[None, :, :, :], Y_new)
        log_det_cov = tf.squeeze(log_det_cov)  # squeezes out the leading singleton dimension
        yt_inv_y = tf.squeeze(yt_inv_y)

        # compute the Gaussian metrics (regardless of emission distribution)
        D_ = tf.cast(self.likelihood.D, settings.float_type)
        self.logp_gauss_data = - 0.5 * yt_inv_y  # this will be terms depending only on the data
        self.logp_gauss = - 0.5 * D_ * np.log(2 * np.pi) - 0.5 * log_det_cov + self.logp_gauss_data  # (S, N)

        # compute loglikelihood under the appropriate emission distribution
        if not self.likelihood.heavy_tail:
            self.logp_data = self.logp_gauss_data
            self.logp = self.logp_gauss
        else:
            dof = tf.cast(self.likelihood.dof, settings.float_type)
            self.logp_data = - 0.5 * (dof + D_) * tf.log(1.0 + yt_inv_y / dof)
            self.logp = tf.lgamma(0.5 * (dof + D_)) - tf.lgamma(0.5 * dof) - 0.5 * D_ * tf.log(np.pi * dof) \
                        - 0.5 * log_det_cov + self.logp_data  # (S, N)

    def map_predict_density(self, X_new, Y_new):
        sess = self.enquire_session()
        return sess.run([self.logp, self.logp_data, self.logp_gauss, self.logp_gauss_data],
                        feed_dict={self.X_new: X_new, self.Y_new: Y_new})


class FullCovarianceRegression(DynamicCovarianceRegression):
    @params_as_tensors
    def build_prior_KL(self):
        KL = super().build_prior_KL()
        if self.likelihood.approx_wishart:
            p_dist = tf.distributions.Gamma(self.likelihood.p_sigma2inv_conc, rate=self.likelihood.p_sigma2inv_rate)
            q_dist = tf.distributions.Gamma(self.likelihood.q_sigma2inv_rate, rate=self.likelihood.q_sigma2inv_rate)
            self.KL_gamma = tf.reduce_sum(q_dist.kl_divergence(p_dist))
            KL += self.KL_gamma
        return KL

    def map_predict(self, X_new):
        """
        Predict either the covariance matrix (in the Wishart process case) or precision matrix (in the inverse Wishart
        process case).

        :param X_new:
        :return:
        """
        sess = self.enquire_session()
        F = sess.run(self.F_mu_new, feed_dict={self.X_new: X_new})  # (N_new, D, nu)

        # grab other required params
        scale_diag = self.likelihood.scale_diag.read_value(sess)

        # it's probably cleanest to just recompute here in numpy
        AF = scale_diag[:, None] * F  # (N, D, nu)
        affa = np.matmul(AF, np.transpose(AF, [0, 2, 1]))  # (N, D, D)

        if self.likelihood.approx_wishart:
            sigma2inv = self.likelihood.q_sigma2inv_conc.read_value(sess) / self.likelihood.q_sigma2inv_rate.read_value(sess)
            sigma2 = sigma2inv ** -1.0
            if self.likelihood.model_inverse:
                affa = affa + np.diag(sigma2 ** -1.0)
            else:
                affa = affa + np.diag(sigma2)
        return affa


class FactoredCovarianceRegression(DynamicCovarianceRegression):
    @params_as_tensors
    def build_prior_KL(self):
        KL = super().build_prior_KL()
        p_dist = tf.distributions.Gamma(self.likelihood.p_sigma2inv_conc, rate=self.likelihood.p_sigma2inv_rate)
        q_dist = tf.distributions.Gamma(self.likelihood.q_sigma2inv_rate, rate=self.likelihood.q_sigma2inv_rate)
        self.KL_gamma = tf.reduce_sum(q_dist.kl_divergence(p_dist))
        KL += self.KL_gamma
        return KL

    def map_predict(self, X_new):
        """
        Compute the components needed for prediction: s2_diag, scale, F. It's more efficient to report this since scale
         is (D, K) and F is (T_test, K, K).

        In the Wishart process case, we construct the covariance matrix as:
            S = np.diag(s2_diag) + U * U^T
        where
            U = np.einsum('jk,ikl->ijl', scale, F)

        In the inverse Wishart process case, we construct the precision matrix as:
            S = np.diag(s2_diag ** -1.0) + U * U^T
        where
            U = np.einsum('jk,ikl->ijl', scale, F)

        :param X_new:
        :return:
        """
        sess = self.enquire_session()
        F = sess.run(self.F_mu_new, feed_dict={self.X_new: X_new})  # (N_new, K, nu)
        scale = self.likelihood.scale.read_value(sess)  # (D, K)
        sigma2inv = self.likelihood.q_sigma2inv_conc.read_value(sess) / self.likelihood.q_sigma2inv_rate.read_value(sess)
        sigma2 = sigma2inv ** -1.0
        return sigma2, scale, F


###########################################
#####  Loglikelihood helper function  #####
###########################################


def get_loglikel(model, Xt, Yt):
    minibatch_size = 100
    loglikel_ = 0.0
    loglikel_data_ = 0.0
    gauss_ll_ = 0.0
    gauss_ll_data_ = 0.0
    for mb in range(-(-len(Xt) // minibatch_size)):
        mb_start = mb * minibatch_size
        mb_finish = (mb + 1) * minibatch_size
        Xt_mb = Xt[mb_start:mb_finish, :]
        Yt_mb = Yt[mb_start:mb_finish, :]
        logp, logp_data, logp_gauss, logp_gauss_data = model.map_predict_density(Xt_mb, Yt_mb)  # (N_new,), (N_new,)
        loglikel_ += np.sum(logp)  # simply summing over the log p(Y_n, X_n | F_n^)
        loglikel_data_ += np.sum(logp_data)
        gauss_ll_ += np.sum(logp_gauss)
        gauss_ll_data_ += np.sum(logp_gauss_data)
    return loglikel_, loglikel_data_, gauss_ll_, gauss_ll_data_


#################################################################
#####  custom GP Monitor tasks to track metrics and params  #####
#################################################################


class LoglikelTensorBoardTask(gpmon.BaseTensorBoardTask):
    def __init__(self, file_writer, model, Xt, Yt, summary_name):
        super().__init__(file_writer, model)
        self.Xt = Xt
        self.Yt = Yt
        self._full_ll = tf.placeholder(settings.float_type, shape=())
        self._full_ll_data = tf.placeholder(settings.float_type, shape=())
        self._full_gauss_ll = tf.placeholder(settings.float_type, shape=())
        self._full_gauss_ll_data = tf.placeholder(settings.float_type, shape=())
        self._summary = tf.summary.merge([tf.summary.scalar(summary_name + '_full', self._full_ll),
                                          tf.summary.scalar(summary_name + '_data', self._full_ll_data),
                                          tf.summary.scalar(summary_name + '_gauss_full', self._full_gauss_ll),
                                          tf.summary.scalar(summary_name + '_gauss_data', self._full_gauss_ll_data),
                                          ])

    def run(self, context: gpmon.MonitorContext, *args, **kwargs) -> None:
        loglikel_, loglikel_data_, gauss_ll_, gauss_ll_data_ = get_loglikel(self.model, self.Xt, self.Yt)
        self._eval_summary(context, {self._full_ll: loglikel_, self._full_ll_data: loglikel_data_,
                                     self._full_gauss_ll: gauss_ll_, self._full_gauss_ll_data: gauss_ll_data_})
