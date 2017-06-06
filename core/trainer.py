import logging
import os
import pickle
import numpy as np
import core.constants as cst
from core.misc import frozen
from copy import deepcopy as copy
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import pandas as pd
import matplotlib.pylab as plt
from sklearn.model_selection import GridSearchCV

with open('saves/common_nomass/scaler.pkl', 'rb') as f:
    nomass_scaler = pickle.load(f)

train = pd.read_table('saves/common_nomass/full_training_set.dst',sep=None, names=cst.features_names_xgdb, header=None)
train_label = np.loadtxt('saves/common_nomass/full_training_labels.lbl')
train['prod_mode'] = train_label
test = pd.read_table('saves/common_nomass/full_test_set.dst',sep=None, names=cst.features_names_xgdb, header=None)
test_label = np.loadtxt('saves/common_nomass/full_test_labels.lbl')
target = 'prod_mode'
predictors = [x for x in train.columns if x not in [target]]
#xgtest = xgb.DMatrix(train[predictors].values, label=train[target].values)
bkg = pd.DataFrame(nomass_scaler.transform(np.loadtxt('saves/common_nomass/ZZTo4l.dst')), columns=cst.features_names_xgdb)

def model_training(model_name):
    models_dict = copy(cst.models_dict)
    analyser, model_weights = models_dict[model_name]
    directory, suffix = cst.dir_suff_dict[cst.features_set_selector]

    training_set = np.loadtxt(directory + 'full_training_set.dst')
    training_labels = np.loadtxt(directory + 'full_training_labels.lbl')

    if model_weights:
        weights = np.array([model_weights[int(cat)] for idx, cat in enumerate(training_labels)])
        analyser.fit(training_set, training_labels, weights)
    else:
        analyser.fit(training_set, training_labels)

    analyser.fit = frozen
    analyser.set_params = frozen

    if not os.path.isdir('saves/classifiers'):
        os.makedirs('saves/classifiers')

    with open('saves/classifiers/' + model_name + suffix + '_categorizer.pkl', mode='wb') as f:
        pickle.dump(analyser, f)

def train_xgcd(model_name, early_stopping_rounds=30, cv_folds=5):
    if cst.features_set_selector != 1:
        logging.warning('Trying xgdb with unsupported features set; models will be trained with nomass')
    alg, class_weights = cst.models_dict[model_name]
    suffix = '_nomass'
    xgb_param = alg.get_xgb_params()
    weights = np.array([class_weights[int(cat)] for cat in train_label])
    xgtrain = xgb.DMatrix(train[predictors].values, label=train[target].values, weight=weights)
    cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                      stratified=True, metrics='merror', early_stopping_rounds=early_stopping_rounds, verbose_eval=None)
    alg.set_params(n_estimators=cvresult.shape[0])
    logging.info('Number of boosting rounds optimized')
    alg.fit(train[predictors], train[target], eval_metric='merror', sample_weight=weights)
    logging.info('Model fit')
    with open('saves/classifiers/' + model_name + suffix + '_categorizer.pkl', 'wb') as f:
        pickle.dump(alg, f)


def generate_predictions(model_name):
    directory, suffix = cst.dir_suff_dict[cst.features_set_selector]
    with open('saves/classifiers/' + model_name + suffix + '_categorizer.pkl', mode='rb') as f:
        classifier = pickle.load(f)
    with open(directory + 'scaler.pkl', mode='rb') as f:
        scaler = pickle.load(f)

    if model_name[0] == 'a':
        scaled_dataset = np.loadtxt(directory + 'full_test_set.dst')
        background_dataset = np.loadtxt(directory + 'ZZTo4l.dst')
        results = classifier.predict(scaled_dataset)
        probas = classifier.predict_proba(scaled_dataset)
        bkg_results = classifier.predict(scaler.transform(background_dataset))
    elif model_name[0] == 'x':
        results = classifier.predict(test)
        probas = classifier.predict_proba(test)
        bkg_results = classifier.predict(bkg)


    out_path = 'saves/predictions/' + model_name + suffix
    np.savetxt(out_path + '_predictions.prd', results)
    np.savetxt(out_path + '_probas.prb', probas)
    np.savetxt(out_path + '_bkg_predictions.prd', bkg_results)

def grid_search():
    param_test1 = {
        'max_depth': range(3, 10, 2),
        'min_child_weight': range(1, 6, 2)
    }
    gsearch1 = GridSearchCV(estimator = XGBClassifier(learning_rate=0.1, n_estimators=75, max_depth=5,
    min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
    objective= 'multi:softmax', n_jobs=16),
    param_grid = param_test1, n_jobs=16, iid=False, cv=5, verbose=10)
    gsearch1.fit(train[predictors], train[target])
    print(gsearch1.cv_results_)
    print(gsearch1.best_params_)
    print(gsearch1.best_score_)

def plot_features_importance(model_name, predictors, cv_folds=5, early_stopping_rounds=30):
    alg = cst.models_dict[model_name]

    xgb_param = alg.get_xgb_params()
    xgtrain = xgb.DMatrix(train[predictors].values, label=train[target].values)
    cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds, stratified=True
                      , metrics='merror', early_stopping_rounds=early_stopping_rounds, verbose_eval=2)
    alg.set_params(n_estimators=cvresult.shape[0])
    alg.fit(train[predictors], train[target], eval_metric='merror')

    feat_imp = pd.Series(alg.feature_importances_).sort_values(ascending=False)
    permutation = np.argsort(-alg.feature_importances_)
    ax = feat_imp.plot(kind='bar', title='Feature Importances', tick_label=features_names)
    ax.set_xticklabels(features_names[permutation])
    plt.ylabel('Feature Importance Score')
    plt.show()
    plt.savefig('saves/figs/xgdb_features_importance.png')
