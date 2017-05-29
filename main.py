import logging
import warnings
import numpy as np
import pickle
import os
import core.trainer as ctg
import core.constants as cst
from core.evaluation import content_plot
from core import assertions
from core import legacy
import core.preprocessing as pr
import core.roc_curve as roc
logging.basicConfig(filename='logs', format='%(levelname)s %(asctime)s %(message)s', level=logging.INFO,
                    datefmt='%H:%M:%S')
logging.info('Logger initialized')

if cst.ignore_warnings:
    warnings.filterwarnings('ignore')


if __name__ == "__main__":
    #pr.full_process((5,))
    #pr.get_background_files((0, 3))

    # if not (tests.common_saves_found() and tests.lengths_consistent()):
    #     pr.full_process()
    # if not (tests.common_saves_found() and tests.lengths_consistent()):
    #     raise UserWarning
    #legacy.read_root_files()
    #legacy.get_background_files()
    #legacy.merge_vector_modes()
    #legacy.generate_metrics()

    for plop in [3]:
        cst.features_set_selector = plop
        directory, suffix = cst.dir_suff_dict[cst.features_set_selector]
        for model_name in cst.models_dict.keys():
            logging.info('Studying model ' + model_name + suffix)
            try:
                open('saves/classifiers/' + model_name + suffix + '_categorizer.pkl', 'rb')
            except IOError:
                logging.info('Training model ' + model_name)
                ctg.model_training(model_name)
            try:
                open('saves/predictions/' + model_name + suffix + '_predictions.prd', 'rb')
                open('saves/predictions/' + model_name + suffix + '_bkg_predictions.prd', 'rb')
            except IOError:
                logging.info('Generating predictions for ' + model_name + suffix)
                ctg.generate_predictions(model_name)

            content_plot(model_name)
        logging.info('All models studied with features set ' + suffix)
    roc.main()
