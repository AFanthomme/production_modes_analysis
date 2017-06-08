import logging
import os
import cPickle as pickle
import numpy as np
import core.constants as cst
from core.misc import frozen
from copy import deepcopy as copy
import xgboost as xgb
import pandas as pd



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
    try:
        current_feature_set
    except NameError:
        global current_feature_set
        current_feature_set = -1

    if cst.features_set_selector != current_feature_set:
        logging.info('Recomputing datasets with new feature set')
        prepare_xgdb()

    alg, class_weights = cst.models_dict[model_name]
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


