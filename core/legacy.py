import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.cm as cm
import matplotlib.pyplot as p
import ROOT as r
from root_numpy import tree2array
import logging
import os
from shutil import rmtree
import numpy.lib.recfunctions as rcf

# Common part of the path to retrieve the root files
base_path = '/data_CMS/cms/ochando/CJLSTReducedTree/170222/'

# This is done here to avoid having root anywhere it doesn't need to be
r.gROOT.LoadMacro("libs/cConstants_no_ext.cc")
r.gROOT.LoadMacro("libs/Discriminants_no_ext.cc")

calculated_features = {
'DVBF2j_ME': (r.DVBF2j_ME, ['p_JJVBF_SIG_ghv1_1_JHUGen_JECNominal', 'p_JJQCD_SIG_ghg2_1_JHUGen_JECNominal', 'ZZMass']),
'DVBF1j_ME' : (r.DVBF1j_ME, ['p_JVBF_SIG_ghv1_1_JHUGen_JECNominal', 'pAux_JVBF_SIG_ghv1_1_JHUGen_JECNominal',
                            'p_JQCD_SIG_ghg2_1_JHUGen_JECNominal', 'ZZMass']),
'DWHh_ME': (r.DWHh_ME, ['p_HadWH_SIG_ghw1_1_JHUGen_JECNominal', 'p_JJQCD_SIG_ghg2_1_JHUGen_JECNominal', 'ZZMass']),
'DZHh_ME': (r.DZHh_ME, ['p_HadZH_SIG_ghz1_1_JHUGen_JECNominal', 'p_JJQCD_SIG_ghg2_1_JHUGen_JECNominal', 'ZZMass']),
}

base_features = [
                'nExtraLep', 'nExtraZ', 'nCleanedJetsPt30', 'nCleanedJetsPt30BTagged_bTagSF',
                'p_JJQCD_SIG_ghg2_1_JHUGen_JECNominal', 'p_JQCD_SIG_ghg2_1_JHUGen_JECNominal',
                'p_JJVBF_SIG_ghv1_1_JHUGen_JECNominal', 'p_JVBF_SIG_ghv1_1_JHUGen_JECNominal',
                'pAux_JVBF_SIG_ghv1_1_JHUGen_JECNominal', 'p_HadWH_SIG_ghw1_1_JHUGen_JECNominal',
                'p_HadZH_SIG_ghz1_1_JHUGen_JECNominal', 'ZZMass', 'PFMET'
                ]

production_modes = ['ggH', 'VBFH', 'WminusH', 'WplusH', 'ZH', 'ttH']
event_categories = ['ggH', 'VBFH', 'VH_hadr', 'VH_lept', 'ZH_met', 'ttH']


# These are the physical constants
luminosity = 35.9   # (fb-1), factor 2 because only half of the initial data set used for evaluation
cross_sections = {'ggH': 13.41, 'VBFH': 1.044, 'WminusH': 0.147, 'WplusH': 0.232, 'ZH': 0.668, 'ttH': 0.393,
                  'VH': 0.232, 'VH_lept': 0.232, 'VH_hadr': 0.232, 'bbH': 0.1347, 'ZH_met': 0.668,
                  'ZZTo4l': 1.256}
event_numbers = {'ZH': 376657.21875, 'WplusH': 252870.65625, 'WminusH': 168069.609375, 'ttH': 327699.28125,
                 'ggH': 999738.125, 'VBFH': 1885726.125, 'VH': 252870.65625, 'VH_lept': 252870.65625,
                 'VH_hadr': 252870.65625, 'bbH':327699.28125, 'ZH_met': 376657.21875, 'ZZTo4l': 6670241.5}
backgrounds = ['ZZTo4l']


def remove_fields(a, *fields_to_remove):
    return a[[name for name in a.dtype.names if name not in fields_to_remove]]

def read_root_files():
        directory = 'saves/legacy'
        if os.path.isdir(directory):
            rmtree(directory)
        os.makedirs(directory)
        logging.info('Directory ' + directory + ' created')

        to_retrieve, to_compute, to_remove = (base_features, None, None)

        for prod_mode in production_modes:
            rfile = r.TFile(base_path + prod_mode + '125/ZZ4lAnalysis.root')
            tree = rfile.Get('ZZTree/candTree')

            ref_mask = None

            if prod_mode not in ['WminusH', 'WplusH', 'ZH']:
                data_set = tree2array(tree, branches=to_retrieve, selection=
                            'ZZsel > 90 && 118 < ZZMass && ZZMass < 130')
                weights = tree2array(tree, branches='overallEventWeight', selection=
                            'ZZsel > 90 && 118 < ZZMass && ZZMass < 130')
                nb_events = np.ma.size(data_set, 0)

                mask = np.ones(nb_events).astype(bool)
                if to_compute:
                    new_features = [np.zeros(nb_events) for _ in range(len(to_compute))]
                    keys = []
                    feature_idx = 0
                    for key, couple in to_compute.iteritems():
                        keys.append(key)
                        plop = new_features[feature_idx]
                        feature_expression, vars_list = couple
                        for event_idx in range(nb_events):
                            tmp = feature_expression(*data_set[vars_list][event_idx])
                            if np.isnan(tmp) or np.isinf(tmp) or np.isneginf(tmp):
                                mask[event_idx] = False
                            plop[event_idx] = tmp
                        new_features[feature_idx] = plop
                        feature_idx += 1
                    data_set = rcf.rec_append_fields(data_set, keys, new_features)

                if not np.all(mask):
                    logging.warning('At least one of the calculated features was Inf or NaN')
                if np.any(ref_mask):
                    mask = ref_mask  # mode 0 = _full should give the most restrictive mask
                data_set = data_set[mask]
                weights = weights[mask]

                if to_remove:
                    data_set = remove_fields(data_set, *to_remove)

                np.savetxt(directory + prod_mode + '_training.txt', data_set)
                np.savetxt(directory + prod_mode + '_weights_training.txt', weights)
                logging.info(prod_mode + ' weights, training and test sets successfully stored in saves/' + directory)

            elif prod_mode == 'ZH':
                decay_criteria = {'_lept': ' && genExtInfo > 10 && !(genExtInfo == 12 || genExtInfo == 14 || genExtInfo == 16)',
                                  '_hadr': ' && genExtInfo < 10',
                                  '_met': ' && (genExtInfo == 12 || genExtInfo == 14 || genExtInfo == 16)'}

                for decay in decay_criteria.keys():

                    data_set = tree2array(tree, branches=to_retrieve, selection=
                            'ZZsel > 90 && 118 < ZZMass && ZZMass < 130' + decay_criteria[decay])
                    weights = tree2array(tree, branches='overallEventWeight', selection=
                            'ZZsel > 90 && 118 < ZZMass && ZZMass < 130' + decay_criteria[decay])

                    nb_events = np.ma.size(data_set, 0)
                    mask = np.ones(nb_events).astype(bool)
                    if to_compute:
                        new_features = [np.zeros(nb_events) for _ in range(len(to_compute))]
                        keys = []
                        feature_idx = 0
                        for key, couple in to_compute.iteritems():
                            keys.append(key)
                            plop = new_features[feature_idx]
                            feature_expression, vars_list = couple
                            for event_idx in range(nb_events):
                                tmp = feature_expression(*data_set[vars_list][event_idx])
                                if np.isnan(tmp) or np.isinf(tmp) or np.isneginf(tmp):
                                    mask[event_idx] = False
                                plop[event_idx] = tmp
                            new_features[feature_idx] = plop
                            feature_idx += 1
                        data_set = rcf.rec_append_fields(data_set, keys, new_features)

                    data_set = data_set[mask]
                    weights = weights[mask]

                    if to_remove:
                        data_set = remove_fields(data_set, *to_remove)

                    np.savetxt(directory + prod_mode + decay + '_training.txt', data_set)
                    np.savetxt(directory + prod_mode + decay + '_weights_training.txt', weights)
                    logging.info(prod_mode + decay + ' weights, training and test sets successfully stored in saves/'
                                 + directory)

            else:
                decay_criteria = {'_lept': ' && genExtInfo > 10', '_hadr': ' && genExtInfo < 10',
                                  '_met': ' && (genExtInfo == 12 || genExtInfo == 14 || genExtInfo == 16)'}

                for decay in decay_criteria.keys():

                    data_set = tree2array(tree, branches=to_retrieve, selection=
                            'ZZsel > 90 && 118 < ZZMass && ZZMass < 130' + decay_criteria[decay])
                    weights = tree2array(tree, branches='overallEventWeight', selection=
                            'ZZsel > 90 && 118 < ZZMass && ZZMass < 130' + decay_criteria[decay])

                    nb_events = np.ma.size(data_set, 0)
                    mask = np.ones(nb_events).astype(bool)
                    if to_compute:
                        new_features = [np.zeros(nb_events) for _ in range(len(to_compute))]
                        keys = []
                        feature_idx = 0
                        for key, couple in to_compute.iteritems():
                            keys.append(key)
                            plop = new_features[feature_idx]
                            feature_expression, vars_list = couple
                            for event_idx in range(nb_events):
                                tmp = feature_expression(*data_set[vars_list][event_idx])
                                if np.isnan(tmp) or np.isinf(tmp) or np.isneginf(tmp):
                                    mask[event_idx] = False
                                plop[event_idx] = tmp
                            new_features[feature_idx] = plop
                            feature_idx += 1
                        data_set = rcf.rec_append_fields(data_set, keys, new_features)

                    if not np.all(mask):
                        logging.warning('At least one of the calculated features was Inf or NaN')
                    if np.any(ref_mask):
                        mask = ref_mask  # mode 0 = _full should give the most restrictive mask

                    data_set = data_set[mask]
                    weights = weights[mask]

                    if  to_remove:
                        data_set = remove_fields(data_set, *to_remove)

                    np.savetxt(directory + prod_mode + decay + '_training.txt', data_set)
                    np.savetxt(directory + prod_mode + decay + '_weights_training.txt', weights)
                    logging.info(prod_mode + decay + ' weights, training and test sets successfully stored in saves/'
                                 + directory)


def get_background_files():

        directory = 'saves/legacy'

        to_retrieve, to_compute, to_remove = (base_features, None, None)

        for background in backgrounds:
            rfile = r.TFile(base_path + background + '/ZZ4lAnalysis.root')
            tree = rfile.Get('ZZTree/candTree')

            data_set = tree2array(tree, branches=to_retrieve, selection=
                        'ZZsel > 90 && 118 < ZZMass && ZZMass < 130')
            weights = tree2array(tree, branches='overallEventWeight', selection=
                        'ZZsel > 90 && 118 < ZZMass && ZZMass < 130')
            nb_events = np.ma.size(data_set, 0)

            mask = np.ones(nb_events).astype(bool)

            if to_compute:
                new_features = [np.zeros(nb_events) for _ in range(len(to_compute))]
                keys = []
                feature_idx = 0
                for key, couple in to_compute.iteritems():
                    keys.append(key)
                    plop = new_features[feature_idx]
                    feature_expression, vars_list = couple
                    for event_idx in range(nb_events):
                        tmp = feature_expression(*data_set[vars_list][event_idx])
                        if np.isnan(tmp) or np.isinf(tmp) or np.isneginf(tmp):
                            mask[event_idx] = False
                        plop[event_idx] = tmp
                    new_features[feature_idx] = plop
                    feature_idx += 1
                data_set = rcf.rec_append_fields(data_set, keys, new_features)

            if to_remove:
                data_set = remove_fields(data_set, *to_remove)

            np.savetxt(directory + background + '.dst', data_set)
            np.savetxt(directory + background + '_weights.wgt', weights)
            logging.info(background + ' weights, training and test sets successfully stored in saves/' + directory)

def merge_vector_modes():
    directory = 'saves/legacy'
    for decay in ['_lept', '_hadr']:
        file_list = [directory + mediator + decay for mediator in ['WplusH', 'WminusH', 'ZH']]

        training_set = np.loadtxt(file_list[0] + '_training.txt')
        weights_train = np.loadtxt(file_list[0] + '_weights_training.txt')
        # Rescale the events weights to have common cross_sections & event numbers equal to the ones of WplusH
        for idx, filename in enumerate(file_list[1:]):
            temp_train = np.loadtxt(filename + '_training.txt')
            temp_weights_train = np.loadtxt(filename + '_weights_training.txt')
            temp_weights_train *= event_numbers['WplusH'] / event_numbers[filename.split('/')[-1].split('_')[0]]

            temp_weights_train *= cross_sections[filename.split('/')[-1].split('_')[0]] / cross_sections['WplusH']

            training_set = np.concatenate((training_set, temp_train), axis=0)
            weights_train = np.concatenate((weights_train, temp_weights_train), axis=0)

        np.savetxt(directory + 'VH' + decay + '_training.txt', training_set)
        np.savetxt(directory + 'VH' + decay + '_weights_training.txt', weights_train)

def generate_metrics():
    directory = 'saves/legacy'
    if not os.path.isdir(directory):
        os.mkdir(directory)
    nb_categories = len(event_categories)
    contents_table = np.array((nb_categories, nb_categories))

    real_cat = [1, 2, 3, 5, 4, 1, 0]
    for cat in range(nb_categories):
        plop = np.loadtxt('saves/legacy/' + event_categories[cat] + '_training.txt')
        plop_weights = np.loadtxt('saves/legacy/' + event_categories[cat] + '_weights_training.txt')
        for event_idx in range(len(plop)):
            nocool_identifier = r.categoryMor17(plop[event_idx][base_features])
            contents_table[real_cat[nocool_identifier], cat] += plop_weights[event_idx]


    bkg_weights = np.loadtxt('saves/legacy/' + 'ZZTo4l_weights.wgt')
    plop = np.loadtxt('saves/legacy/' + 'ZZTo4l.dst')
    bkg_weights *= cross_sections['ZZTo4l'] / event_numbers['ZZTo4l']
    bkg_predictions = [r.categoryMor17(plop[event_idx][base_features]) for event_idx in range(len(bkg_weights))]
    bkg_repartition = np.array([np.sum(bkg_weights[np.where(real_cat[bkg_predictions] == cat)])
                                for cat in range(nb_categories)])

    contents_table *= luminosity
    correct_in_cat = [contents_table[cat, cat] for cat in range(nb_categories)]
    wrong_in_cat = np.sum(np.where(np.logical_not(np.identity(nb_categories, dtype=bool)), contents_table, 0), axis=1)
    cat_total_content = np.sum(contents_table, axis=0)
    purity = [1. / (1. + (bkg_repartition[cat] + wrong_in_cat[cat]) / correct_in_cat[cat]) for cat in range(nb_categories)]
    acceptance = [correct_in_cat[cat] / cat_total_content[cat] for cat in range(nb_categories)]

    print('Purity, acceptance : ' + str(np.mean(purity)[1:]) + ', ' + str(np.mean(acceptance[1:])))

if __name__ == '__main__':
    read_root_files()
    get_background_files()
    merge_vector_modes()
    generate_metrics()