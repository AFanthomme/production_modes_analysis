# import warnings
# warnings.warn("the spam module is deprecated", DeprecationWarning,
#               stacklevel=2)
#
# import pandas as pd
# import warnings
# warnings.filterwarnings('ignore')
# import numpy as np
# import xgboost as xgb
# from xgboost.sklearn import XGBClassifier
# from sklearn import cross_validation, metrics   #Additional scklearn functions
# from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score   #Perforing grid search
# import logging
# from matplotlib.pylab import rcParams
# rcParams['figure.figsize'] = 12, 4
# import os
# import time
# import pickle
# os.environ['PATH'] = os.environ['PATH'] + ';C:\\Program Files\\mingw-w64\\x86_64-5.3.0-posix-seh-rt_v4-rev0\\mingw64\\bin'
#
#
#
# event_categories = np.array(['ggH', 'VBFH', 'VH_hadr', 'VH_lept','ZH_met', 'ttH', 'bbH'])
#
#
#
#
# def sampler(size, n_features=10):
#     if size < 10000 * n_features:
#         return size / n_features
#     else:
#         return 10000
#
# def main():
#     pass
#     # global target, train, predictors, xgb1
#     # train = pd.read_table('saves/common_nomass/full_training_set.dst',sep=None, names=features_names, header=None)
#     # train_label = np.loadtxt('saves/common_nomass/full_training_labels.lbl')
#     # train['prod_mode'] = train_label
#     # grouped = train.groupby('prod_mode')
#     # grouped_bis = grouped.apply(lambda x: x.sample(n=sampler(x.size)))
#     # target = 'prod_mode'
#
#     # predictors = [x for x in train.columns if x not in [target]]
#
#     #train = grouped_bis
#
#     t1 = time.time()
#     modelfit(xgb1, train, predictors)
#     print('total time for all rounds :' + str(time.time() - t1) + 's')
#
#
#
#
#
# if __name__ == '__main__':
#    logging.basicConfig(filename='logs', format='%(levelname)s %(asctime)s %(message)s', level=logging.INFO,
#                     datefmt='%H:%M:%S')
#    logging.info('Logger initialized from xGDB')
#
#    main()
#
#    with warnings.catch_warnings():
#        gsearch1 = GridSearchCV(estimator = XGBClassifier(learning_rate=0.1, n_estimators=75, max_depth=5,
#         min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
#         objective= 'multi:softmax', n_jobs=16),
#         param_grid = param_test1, n_jobs=16, iid=False, cv=5, verbose=10)
#     #   gsearch1.fit(train[predictors], train[target])
#        print(gsearch1.cv_results_)
#        print(gsearch1.best_params_)
#        print(gsearch1.best_score_)


