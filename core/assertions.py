import os
import logging
import pickle
import numpy as np


def common_saves_found():
    directories = ['saves/common_full/', 'saves/common_nodiscr/', 'saves/common_onlydiscr/']
    file_list = ['full_test_set.txt', 'full_training_set.txt']

    for directory in directories:
        for name in file_list:
            if not os.path.isfile(directory + name):
                logging.info(directory + name + ' not found')
                return False
    return True

def lengths_consistent():
    expectations = {'saves/common_full/': 17, 'saves/common_nodiscr/':13, 'saves/common_onlydiscr/':10}
    name = 'full_test_set.txt'

    for directory in expectations.keys():
        if not np.ma.size(np.loadtxt(directory + name), 1) == expectations[directory]:
            logging.info(directory + name + ' has inconsistent number of features.')
            return False
    return True



# def main():
#     model_name = 'adaboost_stumps_300_purity_nodiscr'
#     directory, suffix = dir_suff_dict[features_set_selector]
#     scaled_dataset = np.loadtxt(directory + 'full_test_set.txt')
#
#     with open('saves/' + model_name + suffix + '/categorizer.pkl', mode='rb') as file:
#         classifier = pickle.load(file)
#
#     out_path = 'saves/' + model_name + suffix
#     probas = classifier.predict_proba(scaled_dataset)
#     np.savetxt(out_path + '/probas.txt', probas)
#
# main()


