import os
import json

import numpy as np
import gpflow
import gpflow.training.monitor as gpmon

from .models import FullCovarianceRegression, FactoredCovarianceRegression, LoglikelTensorBoardTask
from .likelihoods import FullCovLikelihood, FactoredCovLikelihood


### Run this as a script

root_savedir = './savedir'
root_logdir = os.path.join(root_savedir, 'tf_logs')

if not os.path.exists(root_savedir):
    os.makedirs(root_savedir)

if not os.path.exists(root_logdir):
    os.makedirs(root_logdir)


#################################
#####  Make a fake dataset  #####
#################################

N = 50  # time points
D = 10
X = np.linspace(0, 5, N)[:, None]  # input time points
Y = np.random.randn(N, D) * 0.2

# holdout a test set
X_train, X_test = X[:40, :], X[40:, :]
Y_train, Y_test = Y[:40, :], Y[40:, :]

# holdout a validation set
holdout_ratio = 0.1
n_train = int(len(X_train) * (1.0 - holdout_ratio))
X_valid, X_train = X_train[n_train:, :], X_train[:n_train, :]
Y_valid, Y_train = Y_train[n_train:, :], Y_train[:n_train, :]


##########################################
#####  Build the GPflow model/graph  #####
##########################################


n_inducing = 10  # number of inducing points
n_samples = 2  # number of Monte Carlo samples
minibatch_size = 16  # minibatch size for training

factored = False  # whether or not to use a factored model
n_factors = None  # number of factors in a factored model (ignored if factored==False)
heavy_tail = False  # whether to use the heavy-tailed emission distribution
model_inverse = True  # if True, then use an inverse Wishart process; if False, use a Wishart process
approx_wishart = True  # if True, use the additive white noise model


# initilize the variational inducing points
x_min = X_train.min()
x_max = X_train.max()
# Z = np.linspace(x_min, x_max, self.n_inducing)[:, None]
Z = x_min + np.random.rand(n_inducing) * (x_max - x_min)
Z = Z[:, None]

# follow the gpflow monitor tutorial to log the optimization procedure
with gpflow.defer_build():

    kern = gpflow.kernels.Matern32(1)

    if not factored:
        likel = FullCovLikelihood(D, n_samples,
                                  heavy_tail=heavy_tail,
                                  model_inverse=model_inverse,
                                  approx_wishart=approx_wishart,
                                  nu=None)

        model = FullCovarianceRegression(X_train, Y_train, kern, likel, Z, minibatch_size=minibatch_size)

    else:
        likel = FactoredCovLikelihood(D, n_samples, n_factors,
                                      heavy_tail=heavy_tail,
                                      model_inverse=model_inverse,
                                      nu=None)

        model = FactoredCovarianceRegression(X_train, Y_train, kern, likel, Z, minibatch_size=minibatch_size)

# print(model.as_pandas_table())


####################################################
#####  GP Monitor tasks for tracking progress  #####
####################################################


# See GP Monitor's demo webpages for more information
monitor_lag = 10  # how often GP Monitor should display training statistics
save_lag = 100  # Don't make this too small. Saving is very I/O intensive

# create the global step parameter tracking the optimization, if using GP monitor's 'create_global_step'
# helper, this MUST be done before creating GP monitor tasks
session = model.enquire_session()
global_step = gpmon.create_global_step(session)

# create the gpmonitor tasks
print_task = gpmon.PrintTimingsTask().with_name('print') \
    .with_condition(gpmon.PeriodicIterationCondition(monitor_lag)) \
    .with_exit_condition(True)

savedir = os.path.join(root_savedir, 'monitor-saves')
saver_task = gpmon.CheckpointTask(savedir).with_name('saver') \
    .with_condition(gpmon.PeriodicIterationCondition(save_lag)) \
    .with_exit_condition(True)

file_writer = gpmon.LogdirWriter(root_logdir, session.graph)

model_tboard_task = gpmon.ModelToTensorBoardTask(file_writer, model).with_name('model_tboard') \
    .with_condition(gpmon.PeriodicIterationCondition(monitor_lag)) \
    .with_exit_condition(True)

train_tboard_task = LoglikelTensorBoardTask(file_writer, model, X_train, Y_train,
                                            summary_name='train_ll').with_name('train_tboard') \
    .with_condition(gpmon.PeriodicIterationCondition(monitor_lag)) \
    .with_exit_condition(True)

# put the tasks together in a monitor
monitor_tasks = [print_task, model_tboard_task, train_tboard_task, saver_task]

# add one more if there is a validation set
if X_valid is not None:
    test_tboard_task = LoglikelTensorBoardTask(file_writer, model, X_valid, Y_valid,
                                               summary_name='test_ll').with_name('test_tboard') \
        .with_condition(gpmon.PeriodicIterationCondition(monitor_lag)) \
        .with_exit_condition(True)
    monitor_tasks.append(test_tboard_task)



##################################
#####  Run the optimization  #####
##################################

learning_rate = 0.01
n_iterations = 100

# create the optimizer
optimiser = gpflow.train.AdamOptimizer(learning_rate)  # create the optimizer

# run optimization steps in the GP Monitor context
with gpmon.Monitor(monitor_tasks, session, global_step, print_summary=True) as monitor:
    optimiser.minimize(model, step_callback=monitor, maxiter=n_iterations, global_step=global_step)


#############################
#####  Demo prediction  #####
#############################


# the format of the predictions depends on whether you're using a factored model; see the definitions in 'models'
if not factored:
    preds = model.map_predict(X_test)
    preds_json = dict(pred_mat=[preds[t, :, :].tolist() for t in range(preds.shape[0])])

else:
    sigma2, scale, F = model.map_predict(X_test)
    preds_json = dict(F=F.tolist(), scale=scale.tolist(), sigma2=sigma2.tolist())


# save predictions in json format
with open(os.path.join(root_savedir, 'preds.json'), 'w') as f:
    json.dump(preds_json, f)