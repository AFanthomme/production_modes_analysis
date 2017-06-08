import logging
import warnings
import core.trainer as ctg
import core.constants as cst
import core.evaluation as evl
import core.preprocessing as pr
import core.roc_curve as roc


logging.basicConfig(filename='logs', format='%(levelname)s %(asctime)s %(message)s', level=logging.INFO,
                    datefmt='%H:%M:%S')
logging.info('Logger initialized from main script')

if cst.ignore_warnings:
    warnings.filterwarnings('ignore')


if __name__ == "__main__":
    pr.get_background_files((0, 1, 2,))

    try:
        #raise IOError
        open('saves/common_nomass/full_test_set.dst')
    except IOError:
        logging.info('Preprocessing datasets (might take some time)')
        pr.full_process((0, 2,))

    for plop in [1]:
        cst.features_set_selector = plop
        directory, suffix = cst.dir_suff_dict[cst.features_set_selector]
        for model_name in cst.models_dict.keys():
            logging.info('Studying model ' + model_name + suffix)
            try:
                #raise IOError
                open('saves/classifiers/' + model_name + suffix + '_categorizer.pkl', 'rb')
            except IOError:
                logging.info('Training model ' + model_name)
                if model_name[0] == 'a':
                    ctg.model_training(model_name)
                elif model_name[0] == 'x':
                    ctg.train_xgcd(model_name)
            try:
                #raise IOError
                open('saves/predictions/' + model_name + suffix + '_predictions.prd', 'rb')
                open('saves/predictions/' + model_name + suffix + '_bkg_predictions.prd', 'rb')
            except IOError:
                logging.info('Generating predictions for ' + model_name + suffix)
                ctg.generate_predictions(model_name)
            try:
                #raise IOError
                open('saves/metrics/' + model_name + suffix + '_acceptance.txt', 'rb')
            except IOError:
                logging.info('Generating metrics for ' + model_name + suffix)
                evl.calculate_metrics(model_name)
        logging.info('All models studied with features set ' + suffix)
