"""
Controls the execution

This script calls the functions defined in the core/ package
To force recompute,

"""
import logging
import warnings
import core.trainer as trn
import core.constants as cst
import core.evaluation as evl
import core.preprocessing as pr
import argparse


parser = argparse.ArgumentParser(description='To control how much recomputing to do')
parser.add_argument('--preproc', action='store_true')
parser.add_argument('--dispatch', action='store_true')
parser.add_argument('--layers', action='store_true')
parser.add_argument('--metrics', action='store_true')
args = parser.parse_args()

warnings.filterwarnings('ignore')

logging.basicConfig(filename='logs', format='%(levelname)s %(asctime)s %(message)s', level=logging.INFO,
                    datefmt='%H:%M:%S')
logging.info('Logger initialized from main script')



evl.content_plot('xgb_ref', save=True, layer=0)
evl.content_plot('xgb_ref_stacked', save=True, layer=1)
evl.content_plot('xgb_ref_doublestacked', save=True, layer=2)

exit()

try:
    open('saves/common_nomass/full_test_set.dst')
except IOError:
    logging.info('Preprocessing datasets (some datasets are longer than others to process)')
    pr.full_process((0, 1,), m_range=cst.mass_range) # Here, can preprocess more feature sets

# Can train our stuff on one of the predefined datasets
for feature_set in [1]:
    cst.features_set_selector = feature_set
    directory, suffix = cst.dir_suff_dict[cst.features_set_selector]
    for model_name in cst.models_dict.keys():
        logging.info('Studying model ' + model_name + suffix)
        try:
            if args.dispatch:
                raise IOError
            open('saves/classifiers/' + model_name + suffix + '_basecategorizer.pkl', 'rb')
        except IOError:
            logging.info('Training model ' + model_name)
            trn.train_xgb(model_name)
        try:
            if args.layers:
                raise IOError
            open('saves/classifiers/' + model_name + suffix + '_subcategorizer6.pkl', 'rb')
            open('saves/classifiers/' + model_name + suffix + '_subsubcategorizer6.pkl', 'rb')
        except IOError:
            logging.info('Training second layer for ' + model_name + suffix)
            trn.train_second_layer(model_name)
            logging.info('Stacking second layer for ' + model_name + suffix)
            trn.make_stacked_predictors(model_name)
            logging.info('Training third layer for ' + model_name + suffix)
            trn.train_third_layer(model_name)
            logging.info('Stacking third layer for ' + model_name + suffix)
            trn.make_double_stacked_predictors(model_name)

        try:
            if args.metrics:
                raise IOError
            open('saves/metrics/' + model_name + suffix + '_acceptance.txt', 'rb')
        except IOError:
            logging.info('Generating metrics for ' + model_name + suffix)
            evl.calculate_metrics(model_name)
        evl.make_pretty_table(model_name)
        evl.content_plot(model_name, True)
    logging.info('All models studied with features set ' + suffix)


evl.content_plot('xgb_ref', save=True, layer=0)
evl.content_plot('xgb_ref_stacked', save=True, layer=1)
evl.content_plot('xgb_ref_doublestacked', save=True, layer=2)
evl.make_pretty_table('xgb_ref_doublestacked')

exit()

