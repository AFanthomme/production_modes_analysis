import logging
import os
import cPickle as pickle
import numpy as np
import core.constants as cst
from copy import deepcopy as copy
import xgboost as xgb
import pandas as pd
from xgboost.sklearn import XGBClassifier

train, test, predictors, target, current_feature_set, bkg_train, bkg_test, train_label, test_label = \
    tuple([None for _ in range(9)])

def prepare_xgdb():
    global train, test, predictors, target, bkg_train, bkg_test, current_feature_set, train_label, test_label
    directory, suffix = cst.dir_suff_dict[cst.features_set_selector]
    features_names_xgdb = cst.features_names_xgdb
    if cst.features_set_selector == 2:
        features_names_xgdb = np.append(features_names_xgdb, ['Z1_Flav', 'Z2_Flav'])

    train = pd.read_table(directory + 'full_training_set.dst',sep=None, names=features_names_xgdb, header=None)
    train_label = np.loadtxt(directory + 'full_training_labels.lbl')
    train['prod_mode'] = train_label
    train.set_index('prod_mode')
    assert train_label.shape[0] == train.shape[0]
    test = pd.read_table(directory + 'full_test_set.dst',sep=None, names=features_names_xgdb, header=None)
    test_label = np.loadtxt(directory + 'full_test_labels.lbl')
    target = 'prod_mode'
    with open(directory + 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    predictors = [x for x in train.columns if x not in [target]]
    bkg = pd.DataFrame(scaler.transform(np.loadtxt(directory + 'ZZTo4l.dst')), columns=features_names_xgdb)
    nb_bkg = bkg.shape[0]
    print(str(nb_bkg) + ' bkg events')
    bkg_train = bkg[nb_bkg // 2:]
    print(str(bkg_train.shape[0]))
    bkg_test = bkg[:nb_bkg // 2]
    print(str(bkg_test.shape[0]))
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
    with open('saves/classifiers/' + model_name + suffix + '_basecategorizer.pkl', 'wb') as f:
        pickle.dump(alg, f)


def generate_predictions(model_name):
    directory, suffix = cst.dir_suff_dict[cst.features_set_selector]

    with open('saves/classifiers/' + model_name + suffix + '_basecategorizer.pkl', mode='rb') as f:
        classifier = pickle.load(f)

    out_path = 'saves/predictions/' + model_name + suffix
    results = classifier.predict(test)
    probas = classifier.predict_proba(test)
    bkg_results = classifier.predict(bkg)
    np.savetxt(out_path + '_predictions.prd', results)
    np.savetxt(out_path + '_probas.prb', probas)
    np.savetxt(out_path + '_bkg_predictions.prd', bkg_results)

    results = classifier.predict(train)
    bkg_results = classifier.predict(bkg_train)
    np.savetxt(out_path + '_train_predictions.prd', results)
    np.savetxt(out_path + '_train_bkg_predictions.prd', bkg_results)


def train_second_layer(model_name, early_stopping_rounds=30, cv_folds=5):
    directory, suffix = cst.dir_suff_dict[cst.features_set_selector]
    _, class_weights_temp = cst.models_dict[model_name]
    class_weights = copy(class_weights_temp)

    if cst.features_set_selector != current_feature_set:
        logging.info('Reloading datasets with new feature set')
        prepare_xgdb()
        logging.info('	New dataset loaded')

    with open('saves/classifiers/' + model_name + suffix + '_basecategorizer.pkl', mode='rb') as f:
        base = pickle.load(f)
    
    predictions = base.predict(train[predictors])
    bkg_predictions = base.predict(bkg_train[predictors])

    for category in [3,]:
        sub_train = train.iloc[np.where(predictions == category)]
        assert sub_train.shape[0] == train_label[np.where(predictions == category)].shape[0]
        sub_label = (train_label[np.where(predictions == category)] == category).astype(int)
        assert sub_train.shape[0] == sub_label.shape[0]
        #sub_wgt = np.array([class_weights[int(cat)] for cat in sub_label])
        sub_wgt = np.array([1 for cat in sub_label])
        
        np.append(sub_train, bkg_train.iloc[np.where(bkg_predictions == category)])
        np.append(sub_label, np.zeros(bkg_train.iloc[np.where(bkg_predictions == category)].shape[0]))
        np.append(sub_wgt, 1. * np.ones(bkg_train.iloc[np.where(bkg_predictions == category)].shape[0]))
        assert sub_train.shape[0] == sub_label.shape[0]
        assert sub_train.shape[0] == sub_wgt.shape[0]
        sub_train['prod_mode'] = sub_label
        threshold = float(np.sum(sub_label == 0)) / float(np.ma.size(sub_label, 0))
        assert np.ma.size(threshold) == 1

        prototype = XGBClassifier(
            learning_rate=0.3,
            n_estimators=1000,
            max_depth=4,
            min_child_weight=4,
            gamma=0,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            eval_metric='error@' + str(threshold),
           # eval_metric='auc',
            n_jobs=16,
        )

        directory, suffix = cst.dir_suff_dict[cst.features_set_selector]
        alg = copy(prototype)
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(sub_train[predictors].values, label=sub_train[target].values, weight=sub_wgt)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          stratified=False, metrics='error@' + str(threshold),
                          early_stopping_rounds=early_stopping_rounds, verbose_eval=5)
        alg.set_params(n_estimators=cvresult.shape[0])
        logging.info('Number of boosting rounds optimized')
        alg.fit(sub_train[predictors], sub_train[target], eval_metric='error@' + str(threshold), sample_weight=sub_wgt)
        logging.info('Model fit')
        print(np.bincount(alg.predict(sub_train[predictors])))
        print(np.bincount(alg.predict(test)))
        with open('saves/classifiers/' + model_name + suffix + '_subcategorizer' + str(category) + '.pkl', 'wb') as f:
            pickle.dump(alg, f)
        
class dummy_predictor(object):
    def __init__(self, output):
        self.output = output
    def predict(self, events):
        return self.output * np.ones(events.shape[0])


class stacked_model(object):
    def __init__(self, base, subs):
        self.base_classifier = base
        self.subclassifiers = subs

    def predict(self, events):
        predictions = np.array(self.base_classifier.predict(events))
        
        for idx, subclassifier in enumerate(self.subclassifiers):
            indices = np.where(predictions.astype(int) == idx)
            subset = events.iloc[indices]
            modify = np.logical_not(subclassifier.predict(subset).astype(bool))
           # print(subset.shape, modify.shape, len(indices[0]))
            print(np.sum(modify), indices[0][modify].shape)
            predictions[indices[0][modify]] = 0

        return predictions


def make_stacked_predictors(model_name):
    directory, suffix = cst.dir_suff_dict[cst.features_set_selector]

    with open('saves/classifiers/' + model_name + suffix + '_basecategorizer.pkl', mode='rb') as f:
        base_classifier = pickle.load(f)

    subclassifiers = [dummy_predictor(1) for cat in range(len(cst.event_categories))]

    for category in [3,]:
        with open('saves/classifiers/' + model_name + suffix + '_subcategorizer' + str(category) + '.pkl', 'rb') as f:
            alg = pickle.load(f)
        subclassifiers[category] = alg

    stacked = stacked_model(base_classifier, subclassifiers)
    with open('saves/classifiers/' + model_name + suffix + '_stacked_categorizer.pkl', mode='wb') as f:
        pickle.dump(stacked, f)

    out_path = 'saves/predictions/' + model_name + suffix + '_stacked'
    results = stacked.predict(test)
    compare = base_classifier.predict(test)
    bkg_results = stacked.predict(bkg_test)
    np.savetxt(out_path + '_predictions.prd', results)
    np.savetxt(out_path + '_bkg_predictions.prd', bkg_results)



