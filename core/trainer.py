"""
Train the three-layer models and generate their predictions.
Other lower layers can be specified if careful.

"""
import logging
import cPickle as pickle
import numpy as np
import core.constants as cst
from copy import deepcopy as copy
import xgboost as xgb
import pandas as pd
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import ParameterGrid


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
    """
    Sets data as global variables.

    Slightly reduces execution time
    :return: Nothing, but sets global values.
    """

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
    bkg_train = pd.DataFrame(np.loadtxt(directory + 'ZZTo4l_training.dst'), columns=features_names_xgdb)
    bkg_train_weights = np.loadtxt(directory + 'full_training_weights.wgt')
    bkg_test = pd.DataFrame(np.loadtxt(directory + 'ZZTo4l_test.dst'), columns=features_names_xgdb)
    current_feature_set = cst.features_set_selector


def train_xgb(model_name, early_stopping_rounds=30, cv_folds=5):
    """
    We make use of some particularities of xgb models,
    :param model_name:
    :param early_stopping_rounds:
    :param cv_folds:
    :return:
    """

    if cst.features_set_selector != current_feature_set:
        logging.info('Reloading datasets with new feature set')
        prepare_xgdb()
        logging.info('New dataset loaded')

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


def significance_factory(event_weights):
    """
    Return the evaluation metrics for the grid-search

    This is necessary since the metrics depends on the considered events and their weights
    WARNING : this might be a problem if trying to modify the code.

    :param event_weights:
    :return:
    """
    try:
        event_weights = event_weights.values
    except AttributeError:
        event_weights = np.array(event_weights)
    def stat_significance_score(estimator, X, y):
        predictions = estimator.predict(X).astype(int)
        signal_in_cat = np.sum(event_weights[np.where(np.logical_and(predictions == 1, y == 1))]) * cst.luminosity
        total_in_cat = np.sum(event_weights[np.where(predictions == 1)]) * cst.luminosity
        return float(signal_in_cat) / np.sqrt(total_in_cat)
    return stat_significance_score

class gridSearch_basic(object):
    """
    Grid-search without cross-validation

    Since our evaluation metrics is so shady, it won't be easily integrated into an sklearn GridSearchCV
    """

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
            raise RuntimeError

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
    """
    Add layer to purify the categories and populate the first one.

    :param model_name:
    :param early_stopping_rounds:
    :param cv_folds:
    :return:
    """

    directory, suffix = cst.dir_suff_dict[cst.features_set_selector]
    _, class_weights_temp = cst.models_dict[model_name]

    if cst.features_set_selector != current_feature_set:
        logging.info('Reloading datasets with new feature set')
        prepare_xgdb()
        logging.info('	New dataset loaded')

    with open('saves/classifiers/' + model_name + suffix + '_basecategorizer.pkl', mode='rb') as f:
        base = pickle.load(f)

    predictions = base.predict(train[predictors])

    for category in range(1, 7):
        # Make the training set out of elements dispatched in our category
        # Do not consider background yet.
        sub_train = train.iloc[np.where(predictions == category)]
        sub_label = (train_label[np.where(predictions == category)] == category).astype(int)
        sub_train['prod_mode'] = sub_label
        sub_train['weights'] = train_weights[np.where(predictions == category)]
        validator_specific_metric = significance_factory(sub_train['weights'])

        directory, suffix = cst.dir_suff_dict[cst.features_set_selector]
        alg = copy(lower_layers)
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(sub_train[predictors].values, label=sub_train[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                stratified=False, metrics='auc', early_stopping_rounds=early_stopping_rounds, verbose_eval=None)
        alg.set_params(n_estimators=cvresult.shape[0])
        logging.info('Number of boosting rounds optimized')

        scales = np.linspace(0.3, 1.3, 11)
        params_dict = ParameterGrid({'scale_pos_weight': scales})
        plop = gridSearch_basic(params_dict, alg, sub_train[predictors].values, sub_train[target].values,
                                scoring=validator_specific_metric)
        plop.fit()
        res = plop.best_params_
        logging.info('\tOptimal scale_pos_weight for validator ' + str(category) + ' : ' + str(res))
        alg.set_params(**plop.best_params_)
        alg.fit(sub_train[predictors], sub_train[target])

        with open('saves/classifiers/' + model_name + suffix + '_subcategorizer' + str(category) + '.pkl', 'wb') as f:
            pickle.dump(alg, f)

def train_third_layer(model_name, early_stopping_rounds=30, cv_folds=5):
    """ Train the background rejectors

    In fact, useful only for ggH and VH_lep, but we do it for everyone because why not

    :param model_name:
    :param early_stopping_rounds:
    :param cv_folds:
    :return:
    """

    directory, suffix = cst.dir_suff_dict[cst.features_set_selector]
    _, class_weights_temp = cst.models_dict[model_name]

    if cst.features_set_selector != current_feature_set:
        logging.info('Reloading datasets with new feature set')
        prepare_xgdb()
        logging.info('	New dataset loaded')

    with open('saves/classifiers/' + model_name + suffix + '_stacked_categorizer.pkl', mode='rb') as f:
        first_stack = pickle.load(f)
    
    predictions = first_stack.predict(train[predictors]).astype(int)
    bkg_predictions = first_stack.predict(bkg_train[predictors]).astype(int)

    for category in range(7):
        sub_train = train.iloc[np.where(predictions == category)]
        sub_label = np.ones_like(sub_train)
        sub_wgt = train_weights[np.where(predictions == category)]
        sub_train['prod_mode'] = sub_label
        sub_train['weights'] = sub_wgt

        sub_train_bkg = bkg_train.iloc[np.where(bkg_predictions == category)]
        sub_wgt_bkg = bkg_train_weights[np.where(bkg_predictions == category)]
        sub_train_bkg['prod_mode'] = np.zeros(sub_train_bkg.shape[0])
        sub_train_bkg['weights'] = sub_wgt_bkg

        sub_train_full = pd.concat([sub_train, sub_train_bkg])
        validator_specific_metric = significance_factory(sub_train_full['weights'])


        if len(sub_train_full['prod_mode'].unique()) == 1 or category == 4:
            logging.info('validator ' + str(category) + ' is useless')
            with open('saves/classifiers/' + model_name + suffix + '_subsubcategorizer' + str(category) + '.pkl', 'wb') as f:
                pickle.dump(dummy_predictor(1), f)
            continue
            
        directory, suffix = cst.dir_suff_dict[cst.features_set_selector]
        alg = copy(lower_layers)
        xgb_param = alg.get_xgb_params()
        alg.set_params(scale_pos_weight=0.06)
        
        alg.fit(sub_train_full[predictors], sub_train_full[target])

        xgtrain = xgb.DMatrix(sub_train_full[predictors].values, label=sub_train_full[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
               stratified=False, metrics='auc', early_stopping_rounds=early_stopping_rounds, verbose_eval=None)
        alg.set_params(n_estimators=cvresult.shape[0])
        logging.info('Number of boosting rounds optimized')

        scales = np.append(np.linspace(0., 0.5, 15), np.linspace(1., 10., 10))
        params_dict = ParameterGrid({'scale_pos_weight': scales})
        plop = gridSearch_basic(params_dict, alg, sub_train_full[predictors].values, sub_train_full[target].values,
                                scoring=validator_specific_metric)
        plop.fit()
        res = plop.best_params_
        logging.info('\tOptimal scale_pos_weight for validator ' + str(category) + ' : ' + str(res))

        alg.set_params(**plop.best_params_)
        alg.fit(sub_train_full[predictors], sub_train_full[target])

        with open('saves/classifiers/' + model_name + suffix + '_subsubcategorizer' + str(category) + '.pkl', 'wb') as f:
            pickle.dump(alg, f)
        
class dummy_predictor(object):
    def __init__(self, output):
        self.output = output
    def predict(self, events):
        return self.output * np.ones(events.shape[0])


class stacked_model(object):
    def __init__(self, base, subs, rejectors=None):
        self.base_classifier = base
        self.subclassifiers = subs
        self.rejectors = rejectors

    def predict(self, events):
        predictions = np.array(self.base_classifier.predict(events))

        for idx, subclassifier in enumerate(self.subclassifiers):
            indices = np.where(predictions.astype(int) == idx)
            subset = events.iloc[indices]
            modify = np.logical_not(subclassifier.predict(subset).astype(bool))
            predictions[indices[0][modify]] = 0
        if self.rejectors:
            for idx, rejector in enumerate(self.rejectors):
                indices = np.where(predictions.astype(int) == idx)
                subset = events.iloc[indices]
                modify = np.logical_not(rejector.predict(subset).astype(bool))
                predictions[indices[0][modify]] = 0  # Set it to -1 to implement bkg rejection
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

    stacked = stacked_model(base_classifier, subclassifiers, validators)
    with open('saves/classifiers/' + model_name + suffix + '_stacked_categorizer.pkl', mode='wb') as f:
        pickle.dump(stacked, f)

    out_path = 'saves/predictions/' + model_name
    out_path += '_doublestacked'
    out_path += suffix
    results = stacked.predict(test)
    bkg_results = stacked.predict(bkg_test)
    np.savetxt(out_path + '_predictions.prd', results)
    np.savetxt(out_path + '_bkg_predictions.prd', bkg_results)


