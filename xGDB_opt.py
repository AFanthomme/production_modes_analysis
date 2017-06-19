""" Tune xgboost hyperparameters """
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV
import os
import core.trainer as t
os.environ['PATH'] = os.environ['PATH'] + \
                     ';C:\\Program Files\\mingw-w64\\x86_64-5.3.0-posix-seh-rt_v4-rev0\\mingw64\\bin'

event_categories = np.array(['ggH', 'VBFH', 'VH_hadr', 'VH_lept','ZH_met', 'ttH', 'bbH'])

xgb_base = XGBClassifier(
     learning_rate =0.1,
     n_estimators=1000,
     max_depth=3,
     min_child_weight=3,
     gamma=0,
     subsample=0.8,
     colsample_bytree=0.8,
     objective= 'multi:softmax',
     num_class=7,
     n_jobs=12,
     )

# Start by finding the best number of estimors for given learning rate.  
# This step could be performed after each tuning
def explore_basics(cv_folds=5):
    alg = xgb_base
    xgb_param = alg.get_xgb_params()
    xgtrain = xgb.DMatrix(t.train[t.predictors].values, label=t.train[t.target].values)
    cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                      stratified=True, metrics='merror', early_stopping_rounds=30, verbose_eval=4)
    alg.set_params(n_estimators=cvresult.shape[0])
    print('Optimal nb of rounds for specified learning rate: ' + str(cvresult.shape[0]))
#explore_basics()


# Then, fix lr and n_est to the right values and do a grid search for max_depth and min_child_weight
# This step can be iterated several times with different parameter grids(coarse then refine)
def grid_search():
    param_grid = {'max_depth':[2, 3, 4, 6,], 'min_child_weight':[2, 3, 4, 6, 8]}
    gsearch1 = GridSearchCV(estimator = XGBClassifier(learning_rate=0.1, n_estimators=228, max_depth=3,
    min_child_weight=3, gamma=0, subsample=0.8, colsample_bytree=0.8,
    objective= 'multi:softmax', n_jobs=16, num_class=7),
    param_grid = param_grid, n_jobs=16, iid=False, cv=5, verbose=10)
    gsearch1.fit(t.train[t.predictors], t.train[t.target])
    print(gsearch1.cv_results_)
    print(gsearch1.best_params_)
    print(gsearch1.best_score_)

# grid_search()

# Change min_child_weight and max_depth
xgb_base = XGBClassifier(
     learning_rate =0.1,
     n_estimators=1000,
     max_depth=4,
     min_child_weight=4,
     gamma=0,
     subsample=0.8,
     colsample_bytree=0.8,
     objective= 'multi:softmax',
     num_class=7,
     n_jobs=12,
     )

# explore_basics()

