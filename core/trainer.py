import logging
import os
import cPickle as pickle
import numpy as np
import core.constants as cst
from core.misc import frozen
from copy import deepcopy as copy
import xgboost as xgb
import pandas as pd

train, test, predictors, target, bkg, current_feature_set, train_label, test_label = tuple([None for _ in range(8)])

def prepare_xgdb():
    global train, test, predictors, target, bkg, current_feature_set, train_label, test_label
    directory, suffix = cst.dir_suff_dict[cst.features_set_selector]
    features_names_xgdb = cst.features_names_xgdb
    if cst.features_set_selector == 2:
        features_names_xgdb = np.append(features_names_xgdb, ['Z1_Flav', 'Z2_Flav'])
    train = pd.read_table(directory + 'full_training_set.dst',sep=None, names=features_names_xgdb, header=None)
    train_label = np.loadtxt(directory + 'full_training_labels.lbl')
    train['prod_mode'] = train_label
    test = pd.read_table(directory + 'full_test_set.dst',sep=None, names=features_names_xgdb, header=None)
    test_label = np.loadtxt(directory + 'full_test_labels.lbl')
    target = 'prod_mode'
    with open(directory + 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    predictors = [x for x in train.columns if x not in [target]]
    bkg = pd.DataFrame(scaler.transform(np.loadtxt(directory + 'ZZTo4l.dst')), columns=features_names_xgdb)
    current_feature_set = cst.features_set_selector


def train_xgcd(model_name, early_stopping_rounds=30, cv_folds=5):
    if cst.features_set_selector != current_feature_set:
        logging.info('Reloading datasets with new feature set')
        prepare_xgdb()
        logging.info('	New dataset loaded')

    alg_temp, class_weights_temp = cst.models_dict[model_name]
    alg = copy(alg_temp)
    class_weights = copy(class_weights_temp)
    directory, suffix = cst.dir_suff_dict[cst.features_set_selector]
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

    results = classifier.predict(test)
    probas = classifier.predict_proba(test)
    bkg_results = classifier.predict(bkg)

    out_path = 'saves/predictions/' + model_name + suffix
    np.savetxt(out_path + '_predictions.prd', results)
    np.savetxt(out_path + '_probas.prb', probas)
    np.savetxt(out_path + '_bkg_predictions.prd', bkg_results)


