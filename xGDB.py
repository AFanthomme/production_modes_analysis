'''  '''

import warnings
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score   #Perforing grid search
import logging
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4
import os
import time
import pickle
os.environ['PATH'] = os.environ['PATH'] + ';C:\\Program Files\\mingw-w64\\x86_64-5.3.0-posix-seh-rt_v4-rev0\\mingw64\\bin'
import core.trainer as t


event_categories = np.array(['ggH', 'VBFH', 'VH_hadr', 'VH_lept','ZH_met', 'ttH', 'bbH'])


with open('saves/classifiers/xgb_200_nomass_categorizer.pkl', mode='wb') as f:
    base_candidate = pickle.load(f)

print(base_candidate.n_estimators)

xgb_base = XGBClassifier(
     learning_rate =0.1,
     n_estimators=1000,
     max_depth=3,
     min_child_weight=3,
     gamma=0,
     subsample=0.8,
     colsample_bytree=0.8,
     objective= 'multi:softmax',
     num_class=8,
     n_jobs=12,
     )

param_grid = {}
gsearch1 = GridSearchCV(estimator = XGBClassifier(learning_rate=0.1, n_estimators=75, max_depth=5,
min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
objective= 'multi:softmax', n_jobs=16),
param_grid = param_grid, n_jobs=16, iid=False, cv=5, verbose=10)
#   gsearch1.fit(train[predictors], train[target])
print(gsearch1.cv_results_)
print(gsearch1.best_params_)
print(gsearch1.best_score_)


