import logging
import warnings
import numpy as np
import pickle
import os
import core.trainer as ctg
import core.constants as cst
import core.evaluation as evl
from core import assertions
from core import legacy
import core.preprocessing as pr
import core.roc_curve as roc
import sys
from contextlib import contextmanager
logging.basicConfig(filename='logs', format='%(levelname)s %(asctime)s %(message)s', level=logging.INFO,
                    datefmt='%H:%M:%S')
logging.info('Logger initialized')

if cst.ignore_warnings:
    warnings.filterwarnings('ignore')

@contextmanager
def redirected(stdout):
    saved_stdout = sys.stdout
    sys.stdout = open(stdout, 'w')
    yield
    sys.stdout = saved_stdout


if __name__ == "__main__":

    evl.content_plot('xgb_200', save=True)
    raise IOError
    try:
        # raise IOError
        open('saves/common_nomass/full_test_set.dst')
    except IOError:
        logging.info('Preprocessing datasets (might take some time)')
        pr.full_process((0, 1,))
    with redirected(stdout='logs'):
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
