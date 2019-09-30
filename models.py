import numpy as np
from scipy.special import logsumexp
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

        :param X:
        :param Y:
        :param kern:
        :param likelihood:
        :param Z:
        :param minibatch_size:
        :param whiten:
        """
        cov_dim = likelihood.cov_dim
        nu = likelihood.nu

        # X and Y are the training dataset, demean and save for prediction time
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
        self.Y_mean = Y_mean  # used when constructing the predictions
        self.construct_predictive_density()

    @params_as_tensors
    def construct_predictive_density(self):
        D = self.likelihood.D
        self.X_new = tf.placeholder(dtype=settings.float_type, shape=[None, 1])
        self.Y_new = tf.placeholder(dtype=settings.float_type, shape=[None, D])
        self.n_samples = tf.placeholder(dtype=settings.int_type, shape=[])

        # demean
        Y_new = self.Y_new - self.Y_mean

        F_mean, F_var = self._build_predict(self.X_new)  # (N_new, num_latent); e.g., num_latent = D * nu
        N_new = tf.shape(F_mean)[0]
        cov_dim = self.likelihood.cov_dim
        self.F_mean_new = tf.reshape(F_mean, [N_new, cov_dim, -1])  # (N_new, cov_dim, nu)
        self.F_var_new = tf.reshape(F_var, [N_new, cov_dim, -1])

        nu = tf.shape(self.F_mean_new)[-1]
        F_samps = tf.random.normal([self.n_samples, N_new, cov_dim, nu], dtype=settings.float_type) \
                  * (self.F_var_new ** 0.5) + self.F_mean_new
        log_det_cov, yt_inv_y = self.likelihood.make_gaussian_components(F_samps, Y_new)

        # compute the Gaussian metrics
        D_ = tf.cast(self.likelihood.D, settings.float_type)
        self.logp_gauss_data = - 0.5 * yt_inv_y
        self.logp_gauss = - 0.5 * D_ * np.log(2 * np.pi) - 0.5 * log_det_cov + self.logp_gauss_data  # (S, N)

        if not self.likelihood.heavy_tail:
            self.logp_data = self.logp_gauss_data
            self.logp = self.logp_gauss
        else:
            dof = tf.cast(self.likelihood.dof, settings.float_type)
            self.logp_data = - 0.5 * (dof + D_) * tf.log(1.0 + yt_inv_y / dof)
            self.logp = tf.lgamma(0.5 * (dof + D_)) - tf.lgamma(0.5 * dof) - 0.5 * D_ * tf.log(np.pi * dof) \
                        - 0.5 * log_det_cov + self.logp_data  # (S, N)

    def mcmc_predict_density(self, X_new, Y_new, n_samples=100):
        sess = self.enquire_session()
        outputs = sess.run([self.logp, self.logp_data, self.logp_gauss, self.logp_gauss_data],
                           feed_dict={self.X_new: X_new, self.Y_new: Y_new, self.n_samples: n_samples})
        log_S = np.log(n_samples)
        return tuple(map(lambda x: logsumexp(x, axis=0) - log_S, outputs))

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

    def mcmc_predict_matrix(self, X_new, n_samples):

        params = self.predict(X_new)
        mu, s2 = params['mu'], params['s2']
        scale_diag = params['scale_diag']

        N_new, D, nu = mu.shape
        F_samps = np.random.randn(n_samples, N_new, D, nu) * np.sqrt(s2) + mu  # (n_samples, N_new, D, nu)
        AF = scale_diag[:, None] * F_samps  # (n_samples, N_new, D, nu)
        affa = np.matmul(AF, np.transpose(AF, [0, 1, 3, 2]))  # (n_samples, N_new, D, D)

        if self.likelihood.approx_wishart:
            sigma2inv_conc = params['sigma2inv_conc']
            sigma2inv_rate = params['sigma2inv_rate']
            sigma2inv_samps = np.random.gamma(sigma2inv_conc, scale=1.0 / sigma2inv_rate, size=[n_samples, D])  # (n_samples, D)

            if self.likelihood.model_inverse:
                lam = np.apply_along_axis(np.diag, axis=0, arr=sigma2inv_samps)  # (n_samples, D, D)
            else:
                lam = np.apply_along_axis(np.diag, axis=0, arr=sigma2inv_samps ** -1.0)
            affa = affa + lam[:, None, :, :]  # (n_samples, N_new, D, D)
        return affa

    def predict(self, X_new):

        sess = self.enquire_session()
        mu, s2 = sess.run([self.F_mean_new, self.F_var_new], feed_dict={self.X_new: X_new})  # (N_new, D, nu), (N_new, D, nu)
        scale_diag = self.likelihood.scale_diag.read_value(sess)  # (D,)
        params = dict(mu=mu, s2=s2, scale_diag=scale_diag)

        if self.likelihood.approx_wishart:
            sigma2inv_conc = self.likelihood.q_sigma2inv_conc.read_value(sess)  # (D,)
            sigma2inv_rate = self.likelihood.q_sigma2inv_rate.read_value(sess)
            params.update(dict(sigma2inv_conc=sigma2inv_conc, sigma2inv_rate=sigma2inv_rate))

        return params

class FactoredCovarianceRegression(DynamicCovarianceRegression):
    @params_as_tensors
    def build_prior_KL(self):
        KL = super().build_prior_KL()
        p_dist = tf.distributions.Gamma(self.likelihood.p_sigma2inv_conc, rate=self.likelihood.p_sigma2inv_rate)
        q_dist = tf.distributions.Gamma(self.likelihood.q_sigma2inv_rate, rate=self.likelihood.q_sigma2inv_rate)
        self.KL_gamma = tf.reduce_sum(q_dist.kl_divergence(p_dist))
        KL += self.KL_gamma
        return KL

    def predict(self, X_new):
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
        mu, s2 = sess.run([self.F_mean_new, self.F_var_new], feed_dict={self.X_new: X_new})  # (N_new, D, nu), (N_new, D, nu)
        scale = self.likelihood.scale.read_value(sess)  # (D, Kv)

        sigma2inv_conc = self.likelihood.q_sigma2inv_conc.read_value(sess)  # (D,)
        sigma2inv_rate = self.likelihood.q_sigma2inv_rate.read_value(sess)
        params = dict(mu=mu, s2=s2, scale=scale, sigma2inv_conc=sigma2inv_conc, sigma2inv_rate=sigma2inv_rate)
        return params


###########################################
#####  Loglikelihood helper function  #####
###########################################


def get_loglikel(model, Xt, Yt, minibatch_size=100):
    loglikel_ = 0.0
    loglikel_data_ = 0.0
    gauss_ll_ = 0.0
    gauss_ll_data_ = 0.0
    for mb in range(-(-len(Xt) // minibatch_size)):
        mb_start = mb * minibatch_size
        mb_finish = (mb + 1) * minibatch_size
        Xt_mb = Xt[mb_start:mb_finish, :]
        Yt_mb = Yt[mb_start:mb_finish, :]
        logp, logp_data, logp_gauss, logp_gauss_data = model.mcmc_predict_density(Xt_mb, Yt_mb)  # (N_new,), (N_new,)
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
