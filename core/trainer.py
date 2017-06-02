import logging
import os
import pickle
import numpy as np
import core.constants as cst
from core.misc import frozen
from copy import deepcopy as copy

def model_training(model_name):
    models_dict = copy(cst.models_dict)
    analyser, model_weights = models_dict[model_name]

    directory, suffix = cst.dir_suff_dict[cst.features_set_selector]

    training_set = np.loadtxt(directory + 'full_training_set.dst')
    training_labels = np.loadtxt(directory + 'full_training_labels.lbl')
    training_weights = np.loadtxt(directory + 'full_training_weights.wgt')

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


def generate_predictions(model_name):
    directory, suffix = cst.dir_suff_dict[cst.features_set_selector]
    scaled_dataset = np.loadtxt(directory + 'full_test_set.dst')
    background_dataset = np.loadtxt(directory + 'ZZTo4l.dst')

    with open('saves/classifiers/' + model_name + suffix + '_categorizer.pkl', mode='rb') as f:
        classifier = pickle.load(f)
    with open(directory + 'scaler.pkl', mode='rb') as f:
        scaler = pickle.load(f)

    results = classifier.predict(scaled_dataset)
    probas = classifier.predict_proba(scaled_dataset)
    bkg_results = classifier.predict(scaler.transform(background_dataset))

    out_path = 'saves/predictions/' + model_name + suffix
    np.savetxt(out_path + '_predictions.prd', results)
    np.savetxt(out_path + '_probas.prb', probas)
    np.savetxt(out_path + '_bkg_predictions.prd', bkg_results)
