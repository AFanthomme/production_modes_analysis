import logging
import warnings
import core.trainer as ctg
import core.constants as cst
import core.evaluation as evl
import core.preprocessing as pr


logging.basicConfig(filename='logs', format='%(levelname)s %(asctime)s %(message)s', level=logging.INFO,
                    datefmt='%H:%M:%S')
logging.info('Logger initialized from main script')

if cst.ignore_warnings:
    warnings.filterwarnings('ignore')


#pr.get_background_files((0, 1))
#evl.make_pretty_table('xgb_200')

try:
    # raise IOError
    open('saves/common_nomass/full_test_set.dst')
except IOError:
    logging.info('Preprocessing datasets (might take some time)')
    pr.full_process((0, 1,), m_range=cst.mass_range)

for plop in [1]:
    cst.features_set_selector = plop
    directory, suffix = cst.dir_suff_dict[cst.features_set_selector]
    for model_name in cst.models_dict.keys():
        logging.info('Studying model ' + model_name + suffix)
        try:
            #raise IOError
            open('saves/classifiers/' + model_name + suffix + '_basecategorizer.pkl', 'rb')
        except IOError:
            logging.info('Training model ' + model_name)
            if model_name[0] == 'a':
                ctg.model_training(model_name)
            elif model_name[0] == 'x':
                ctg.train_xgcd(model_name)
        try:
            #raise IOError
            open('saves/classifiers/' + model_name + suffix + '_subcategorizer3.pkl', 'rb')
        except IOError:
            logging.info('Training model ' + model_name)
            if model_name[0] == 'a':
                ctg.model_training(model_name)
            elif model_name[0] == 'x':
                logging.info('Training second layer for ' + model_name + suffix)
                ctg.train_second_layer(model_name)
                logging.info('Stacking layers for ' + model_name + suffix)
                ctg.make_stacked_predictors(model_name)
        try:
            #raise IOError
            open('saves/predictions/' + model_name + suffix + '_predictions.prd', 'rb')
            open('saves/predictions/' + model_name + suffix + '_bkg_predictions.prd', 'rb')
        except IOError:
            logging.info('Generating predictions for ' + model_name + suffix)
            #ctg.generate_predictions(model_name)
        try:
            raise IOError
            open('saves/metrics/' + model_name + suffix + '_acceptance.txt', 'rb')
        except IOError:
            logging.info('Generating metrics for ' + model_name + suffix)
            evl.calculate_metrics(model_name)
        evl.make_pretty_table(model_name)
        # evl.content_plot(model_name, True)
    logging.info('All models studied with features set ' + suffix)
#evl.custom_roc()
