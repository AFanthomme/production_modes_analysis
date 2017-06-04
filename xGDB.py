import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score   #Perforing grid search
import logging
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4
import os
import time
import pickle
os.environ['PATH'] = os.environ['PATH'] + ';C:\\Program Files\\mingw-w64\\x86_64-5.3.0-posix-seh-rt_v4-rev0\\mingw64\\bin'
features_names = np.array(['nExtraLep', 'nExtraZ', 'nCleanedJetsPt30', 'nCleanedJetsPt30BTagged_bTagSF',
                  'PFMET', 'DVBF2j_ME', 'DVBF1j_ME', 'DWHh_ME', 'DZHh_ME'])

event_categories = np.array(['ggH', 'VBFH', 'VH_hadr', 'VH_lept','ZH_met', 'ttH', 'bbH'])

def modelfit(alg, dtrain, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds, stratified=True
                          , metrics='merror', early_stopping_rounds=early_stopping_rounds, verbose_eval=2)
        alg.set_params(n_estimators=cvresult.shape[0])
    print('Optimal number of rounds:'+ str( cvresult.shape[0]))
    # Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target], eval_metric='merror')
    print('Fit done')
    # Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]
    
    np.savetxt('saves/predictions/xgb_ref_nomass_predictions.prd', dtrain_predictions)

    # Print model report:
    print "\nModel Report"
    print "Accuracy : %.4g" % metrics.accuracy_score(dtrain['prod_mode'].values, dtrain_predictions)

    with open('saves/classifiers/xgb_ref_nomass_classifier.pkl', 'wb')as f:
        pickle.dump(alg, f)

    feat_imp = pd.Series(alg.feature_importances_).sort_values(ascending=False)
    permutation = np.argsort(-alg.feature_importances_)
    ax = feat_imp.plot(kind='bar', title='Feature Importances', tick_label=features_names)
    ax.set_xticklabels(features_names[permutation])
    plt.ylabel('Feature Importance Score')
    plt.show()
    plt.savefig('saves/figs/xgdb_features_importance.png')

def sampler(size, n_features=10):
    if size < 10000 * n_features:
        return size / n_features
    else:
        return 10000
def main():
    global target, train, predictors, xgb1 
    train = pd.read_table('saves/common_nomass/full_training_set.dst',sep=None, names=features_names, header=None)
    train_label = np.loadtxt('saves/common_nomass/full_training_labels.lbl')
    train['prod_mode'] = train_label
    grouped = train.groupby('prod_mode')
    grouped_bis = grouped.apply(lambda x: x.sample(n=sampler(x.size)))
    target = 'prod_mode' 
    
    predictors = [x for x in train.columns if x not in [target]]
    xgb1 = XGBClassifier(
     learning_rate =0.1,
     n_estimators=1000,
     max_depth=3,
     min_child_weight=3,
     gamma=0,
     subsample=0.8,
     colsample_bytree=0.8,
     objective= 'multi:softmax',
     num_class=8,
     n_jobs=16,
     )
    #train = grouped_bis   

    t1 = time.time() 
    modelfit(xgb1, train, predictors)
    print('total time for all rounds :' + str(time.time() - t1) + 's')


if __name__ == '__main__':
   logging.basicConfig(filename='logs', format='%(levelname)s %(asctime)s %(message)s', level=logging.INFO,
                    datefmt='%H:%M:%S')
   logging.info('Logger initialized from xGDB')
   param_test1 = {
    'max_depth':range(3,10,2),
    'min_child_weight':range(1,6,2)
   }
   main()

   with warnings.catch_warnings():
       gsearch1 = GridSearchCV(estimator = XGBClassifier(learning_rate=0.1, n_estimators=75, max_depth=5,
        min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
        objective= 'multi:softmax', n_jobs=16), 
        param_grid = param_test1, n_jobs=16, iid=False, cv=5, verbose=10)
    #   gsearch1.fit(train[predictors], train[target])
       print(gsearch1.grid_scores_)
       print(gsearch1.best_params_)
       print(gsearch1.best_score_)
     

