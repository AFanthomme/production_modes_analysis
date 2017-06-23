import logging
import os
import cPickle as pickle
import numpy as np
import core.constants as cst
from copy import deepcopy as copy
import xgboost as xgb
import pandas as pd
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import ParameterGrid
from itertools import izip

train, test, predictors, target, current_feature_set, bkg_train, bkg_test, train_label, test_label, \
test_weights, train_weights, bkg_train_weights = \
    tuple([None for _ in range(12)])

lower_layers = XGBClassifier(
    learning_rate=0.05,
    n_estimators=1000,
    scale_pos_weight=1.,
    max_depth=4,
    min_child_weight=4,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    eval_metric='auc',
    n_jobs=16,
)

def prepare_xgdb():
    global train, test, predictors, target, bkg_train, bkg_test, current_feature_set, train_label, test_label
    global test_weights, train_weights, bkg_train_weights
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
    test_weights = np.loadtxt(directory + 'full_test_weights.wgt')
    train_weights = np.loadtxt(directory + 'full_training_weights.wgt')
    target = 'prod_mode'
    predictors = [x for x in train.columns if x not in [target]]
    bkg_train = pd.DataFrame(np.loadtxt(directory + 'ZZTo4ltraining.dst'), columns=features_names_xgdb)
    bkg_train_weights = np.loadtxt(directory + 'full_training_weights.wgt')
    bkg_test = pd.DataFrame(np.loadtxt(directory + 'ZZTo4l_test.dst'), columns=features_names_xgdb)
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
    bkg_results = classifier.predict(bkg_test)
    np.savetxt(out_path + '_predictions.prd', results)
    np.savetxt(out_path + '_probas.prb', probas)
    np.savetxt(out_path + '_bkg_predictions.prd', bkg_results)

    results = classifier.predict(train)
    bkg_results = classifier.predict(bkg_train)
    np.savetxt(out_path + '_train_predictions.prd', results)
    np.savetxt(out_path + '_train_bkg_predictions.prd', bkg_results)

def formatter(model_name):
    fs_labels = ['_4e', '_2e2mu', '_4mu']

    no_care, suffix = cst.dir_suff_dict[cst.features_set_selector]
    model_name += suffix
    suffix += '/'

    true_categories_ref = np.loadtxt('saves/common' + suffix + 'full_test_labels.lbl')
    weights_ref = np.loadtxt('saves/common' + suffix + 'full_test_weights.wgt')
    predictions_ref = np.loadtxt('saves/predictions/' + model_name + '_predictions.prd')
    final_states = np.loadtxt('saves/common' + suffix + 'full_test_finalstates.dst').astype(int)
    bkg_predictions_ref = np.loadtxt('saves/predictions/' + model_name + '_bkg_predictions.prd')
    bkg_weights_ref = np.loadtxt('saves/common' + suffix + 'ZZTo4l_weights_test.wgt')
    bkg_final_states = np.loadtxt('saves/common' + suffix + 'ZZTo4l_finalstates_test.dst').astype(int)
    bkg_weights_ref *= 0.5  # Here there was no train/test split
    nb_categories = len(cst.event_categories)
    nb_processes = nb_categories + 1  # Consider all background at once

    for fs_idx, fs_label in enumerate(fs_labels):
        contents_table = np.zeros((nb_categories, nb_processes))
        mask_fs = np.where(final_states == fs_idx)
        true_categories = true_categories_ref[mask_fs]
        weights = weights_ref[mask_fs]
        predictions = predictions_ref[mask_fs]

        bkg_mask_fs = np.where(bkg_final_states == fs_idx)
        bkg_weights = bkg_weights_ref[bkg_mask_fs]
        bkg_predictions = bkg_predictions_ref[bkg_mask_fs]

        for true_tag, predicted_tag, rescaled_weight in izip(true_categories, predictions, weights):
            contents_table[predicted_tag, true_tag] += rescaled_weight

        for predicted_tag, rescaled_weight in izip(bkg_predictions, bkg_weights):
            contents_table[predicted_tag, -1] += rescaled_weight

        contents_table *= cst.luminosity

   # template.format(contents_table)



# def custom_scoring(estimator, X, y, mode=1):
#     ref_up = [0.22, 1.15, 3.79, 3.2, 2.82, 10.]
#     ref_down = [0.2, 0.88, 1., 1., 1., 1.]
#
#     # Complicated sequence to get the error bars downwards (std_down) and upwards (std_up)
#     std_down, std_up = 0.1, 0.1
#     error = std_down + std_up
#
#     if std_down > ref_down[mode]:
#         error += 3.
#     if std_up > ref_up[mode]:
#         error += 3.
#     return -error


def significance_factory(event_weights):
    def stat_significance_score(estimator, X, y):
        predictions = estimator.predict(X).astype(int)
        signal_in_cat = np.sum(event_weights[np.where(np.logical_and(predictions == 1, y == 1))]) * cst.luminosity
        total_in_cat = np.sum(event_weights[np.where(predictions == 1)]) * cst.luminosity
        return float(signal_in_cat) / np.sqrt(total_in_cat)
    return stat_significance_score

class gridSearch_basic(object):

    def __init__(self, param_grid, estimator, X, y, scoring=None):
        self.param_grid = param_grid
        self.estimator = estimator
        self.X = X
        self.y = y
        self.best_params_ = {}
        self.best_estimator_ = None
        self.best_score_ = float('-inf')
        if scoring:
            self.scorer = scoring
        else:
            exit()

    def fit(self, verbose=True):
        for idx, params in enumerate(self.param_grid):
            estimator = copy(self.estimator)
            estimator.set_params(**params)
            estimator.fit(self.X, self.y)
            tmp = self.scorer(estimator, self.X, self.y)
            if verbose:
                logging.info('\t Params ' + str(idx) + ' over ' + str(len(self.param_grid)) + ' : ' + str(tmp))
            if tmp > self.best_score_:
                self.best_score_ = tmp
                self.best_params_ = params
                self.best_estimator_ = estimator


def train_second_layer(model_name, early_stopping_rounds=30, cv_folds=5):
    directory, suffix = cst.dir_suff_dict[cst.features_set_selector]
    _, class_weights_temp = cst.models_dict[model_name]

    if cst.features_set_selector != current_feature_set:
        logging.info('Reloading datasets with new feature set')
        prepare_xgdb()
        logging.info('	New dataset loaded')

    with open('saves/classifiers/' + model_name + suffix + '_basecategorizer.pkl', mode='rb') as f:
        base = pickle.load(f)

    predictions = base.predict(train[predictors])
    bkg_predictions = base.predict(bkg_train[predictors])

    for category in range(1, 7):
        sub_train = train.iloc[np.where(predictions == category)]
        sub_label = (train_label[np.where(predictions == category)] == category).astype(int)
        sub_wgt = train_weights[np.where(predictions == category)]

        validator_specific_metric = significance_factory(sub_wgt)

        np.append(sub_train, bkg_train.iloc[np.where(bkg_predictions == category)])
        np.append(sub_label, np.zeros(bkg_train.iloc[np.where(bkg_predictions == category)].shape[0]))

        sub_train['prod_mode'] = sub_label

        directory, suffix = cst.dir_suff_dict[cst.features_set_selector]
        alg = copy(lower_layers)
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(sub_train[predictors].values, label=sub_train[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          stratified=False, metrics='auc',
                          early_stopping_rounds=early_stopping_rounds, verbose_eval=15)
        alg.set_params(n_estimators=cvresult.shape[0])
        logging.info('Number of boosting rounds optimized')

        scales = np.linspace(0.3, 1.3, 11)
        params_dict = ParameterGrid({'scale_pos_weight': scales})

        # Here we do a grid search without cv beacause it's painful to implement with our shady metrics
        plop = gridSearch_basic(params_dict, alg, sub_train[predictors].values, sub_train[target].values,
                                scoring=validator_specific_metric)
        plop.fit()
        res = plop.best_params_
        logging.info('\t \tOptimal scale_pos_weight for validator ' + str(category) + ' : ' + str(res))

        alg.set_params(**plop.best_params_)
        alg.fit(sub_train[predictors], sub_train[target])

        with open('saves/classifiers/' + model_name + suffix + '_subcategorizer' + str(category) + '.pkl', 'wb') as f:
            pickle.dump(alg, f)

def train_third_layer(model_name, early_stopping_rounds=30, cv_folds=5):
    directory, suffix = cst.dir_suff_dict[cst.features_set_selector]
    _, class_weights_temp = cst.models_dict[model_name]

    if cst.features_set_selector != current_feature_set:
        logging.info('Reloading datasets with new feature set')
        prepare_xgdb()
        logging.info('	New dataset loaded')

    with open('saves/classifiers/' + model_name + suffix + '_stacked_categorizer.pkl', mode='rb') as f:
        first_stack = pickle.load(f)
    
    predictions = first_stack.predict(train[predictors])
    bkg_predictions = first_stack.predict(bkg_train[predictors])

    for category in range(7):
        sub_train = train.iloc[np.where(predictions == category)]
        # sub_label = (train_label[np.where(predictions == category)] == category).astype(int)
        sub_label = np.ones_like(sub_train)
        sub_wgt = train_weights[np.where(predictions == category)]

        validator_specific_metric = significance_factory(sub_wgt)

        np.append(sub_train, bkg_train.iloc[np.where(bkg_predictions == category)])
        np.append(sub_label, np.zeros(bkg_train.iloc[np.where(bkg_predictions == category)].shape[0]))

        sub_train['prod_mode'] = sub_label

        directory, suffix = cst.dir_suff_dict[cst.features_set_selector]
        alg = copy(lower_layers)
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(sub_train[predictors].values, label=sub_train[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          stratified=False, metrics='auc',
                          early_stopping_rounds=early_stopping_rounds, verbose_eval=15)
        alg.set_params(n_estimators=cvresult.shape[0])
        logging.info('Number of boosting rounds optimized')

        scales = np.linspace(0.3, 1.3, 11)
        params_dict = ParameterGrid({'scale_pos_weight': scales})

        # Here we do a grid search without cv beacause it's painful to implement with our shady metrics
        plop = gridSearch_basic(params_dict, alg, sub_train[predictors].values, sub_train[target].values,
                                scoring=validator_specific_metric)
        plop.fit()
        res = plop.best_params_
        logging.info('\t \tOptimal scale_pos_weight for validator ' + str(category) + ' : ' + str(res))

        alg.set_params(**plop.best_params_)
        alg.fit(sub_train[predictors], sub_train[target])

        with open('saves/classifiers/' + model_name + suffix + '_subsubcategorizer' + str(category) + '.pkl', 'wb') as f:
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
            predictions[indices[0][modify]] = 0

        return predictions


class double_stacked_model(object):
    def __init__(self, base, subs, subsubs):
        self.base_classifier = base
        self.subclassifiers = subs
        self.rejectors = subsubs


    def predict(self, events):
        predictions = np.array(self.base_classifier.predict(events))

        for idx, subclassifier in enumerate(self.subclassifiers):
            indices = np.where(predictions.astype(int) == idx)
            subset = events.iloc[indices]
            modify = np.logical_not(subclassifier.predict(subset).astype(bool))
            predictions[indices[0][modify]] = 0

        for idx, rejector in enumerate(self.rejectors):
            indices = np.where(predictions.astype(int) == idx)
            subset = events.iloc[indices]
            modify = np.logical_not(rejector.predict(subset).astype(bool))
            predictions[indices[0][modify]] = 0  # Maybe set it to a new category for background
        return predictions

def make_stacked_predictors(model_name):
    directory, suffix = cst.dir_suff_dict[cst.features_set_selector]

    with open('saves/classifiers/' + model_name + suffix + '_basecategorizer.pkl', mode='rb') as f:
        base_classifier = pickle.load(f)

    subclassifiers = [dummy_predictor(1) for _ in range(len(cst.event_categories))]

    for category in range(1, 7):
        with open('saves/classifiers/' + model_name + suffix + '_subcategorizer' + str(category) + '.pkl', 'rb') as f:
            alg = pickle.load(f)
        subclassifiers[category] = alg

    stacked = stacked_model(base_classifier, subclassifiers)
    with open('saves/classifiers/' + model_name + suffix + '_stacked_categorizer.pkl', mode='wb') as f:
        pickle.dump(stacked, f)


    out_path = 'saves/predictions/' + model_name
    compare = base_classifier.predict(test)
    bkg_compare = base_classifier.predict(bkg_test)
    np.savetxt(out_path + suffix + '_predictions.prd', compare)
    np.savetxt(out_path + suffix + '_bkg_predictions.prd', bkg_compare)
    
    out_path += '_stacked' + suffix
    results = stacked.predict(test)
    bkg_results = stacked.predict(bkg_test)
    np.savetxt(out_path + '_predictions.prd', results)
    np.savetxt(out_path + '_bkg_predictions.prd', bkg_results)


def make_double_stacked_predictors(model_name):
    directory, suffix = cst.dir_suff_dict[cst.features_set_selector]

    with open('saves/classifiers/' + model_name + suffix + '_basecategorizer.pkl', mode='rb') as f:
        base_classifier = pickle.load(f)

    subclassifiers = [dummy_predictor(1) for _ in range(len(cst.event_categories))]
    validators = [dummy_predictor(1) for _ in range(len(cst.event_categories))]

    for category in range(1, 7):
        with open('saves/classifiers/' + model_name + suffix + '_subcategorizer' + str(category) + '.pkl', 'rb') as f:
            alg = pickle.load(f)
        subclassifiers[category] = alg

    for category in range(7):
        with open('saves/classifiers/' + model_name + suffix + '_subsubcategorizer' + str(category) + '.pkl', 'rb') as f:
            alg = pickle.load(f)
        validators[category] = alg

    stacked = double_stacked_model(base_classifier, subclassifiers, validators)
    with open('saves/classifiers/' + model_name + suffix + '_stacked_categorizer.pkl', mode='wb') as f:
        pickle.dump(stacked, f)

    out_path = 'saves/predictions/' + model_name
    out_path += '_doublestacked'
    out_path += suffix
    results = stacked.predict(test)
    bkg_results = stacked.predict(bkg_test)
    np.savetxt(out_path + '_predictions.prd', results)
    np.savetxt(out_path + '_bkg_predictions.prd', bkg_results)



