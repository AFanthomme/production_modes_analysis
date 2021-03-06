'''
Preprocessing from ROOT to simple numpy datasets

We removed scaling as it is not necessary for Decision Trees. All this could probably be improved significantly
by using pandas
'''
import os
import logging
from shutil import rmtree
import ROOT as r
from root_numpy import tree2array
import numpy as np
import numpy.lib.recfunctions as rcf
import core.constants as cst


# Common part of the path to retrieve the root files
base_path = '/data_CMS/cms/ochando/CJLSTReducedTree/170222/'

# Dictionary of calculated quantities names, function and name of the functions arguments (must be features of the tree)
r.gROOT.LoadMacro("libs/cConstants_no_ext.cc")
r.gROOT.LoadMacro("libs/Discriminants_no_ext.cc")
calculated_features = \
{
'DVBF2j_ME': (r.DVBF2j_ME, ['p_JJVBF_SIG_ghv1_1_JHUGen_JECNominal', 'p_JJQCD_SIG_ghg2_1_JHUGen_JECNominal', 'ZZMass']),
'DVBF1j_ME' : (r.DVBF1j_ME, ['p_JVBF_SIG_ghv1_1_JHUGen_JECNominal', 'pAux_JVBF_SIG_ghv1_1_JHUGen_JECNominal',
                            'p_JQCD_SIG_ghg2_1_JHUGen_JECNominal', 'ZZMass']),
'DWHh_ME': (r.DWHh_ME, ['p_HadWH_SIG_ghw1_1_JHUGen_JECNominal', 'p_JJQCD_SIG_ghg2_1_JHUGen_JECNominal', 'ZZMass']),
'DZHh_ME': (r.DZHh_ME, ['p_HadZH_SIG_ghz1_1_JHUGen_JECNominal', 'p_JJQCD_SIG_ghg2_1_JHUGen_JECNominal', 'ZZMass']),
}

# For each feature selection mode, (to get from root file, to calculate, to remove)
features_specs = [(cst.base_features + ['Z1Flav', 'Z2Flav'], calculated_features, None),
                  (cst.base_features + ['Z1Flav', 'Z2Flav'], calculated_features, cst.likelihood_names + ['ZZMass']
                   + ['Z1Flav', 'Z2Flav']),
                  ]


def remove_fields(labeled_arr, *fields_to_remove):
    return labeled_arr[[name for name in labeled_arr.dtype.names if name not in fields_to_remove]]

def post_selection_processing(data_set, features_tuple):
    """
    Adds calculated features and removes unwanted ones
    :param data_set: numpy structured array
    :param features_tuple: a tuple of labels lists (to retrieve, to compute, to remove)
    :return: The dataset with the new calculated fields and without the removed ones.
    """
    nb_events = np.ma.size(data_set, 0)
    mask = np.ones(nb_events).astype(bool)
    dont_care, to_compute, to_remove = features_tuple
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

    blob1 = data_set[['Z1Flav']].astype(int)
    blob2 = data_set[['Z2Flav']].astype(int)
    final_states = identify_final_state(blob1, blob2)

    if to_remove:
        data_set = remove_fields(data_set, *to_remove)

    return data_set, final_states, mask

def get_background_files(modes=(0, 1, 2), m_range=('118', '130')):
    for features_mode in modes:
        directory, suffix = cst.dir_suff_dict[features_mode]
        to_retrieve, to_compute, to_remove = features_specs[features_mode]
        for background in cst.backgrounds:
            rfile = r.TFile(base_path + background + '/ZZ4lAnalysis.root')
            tree = rfile.Get('ZZTree/candTree')

            data_set = tree2array(tree, branches=to_retrieve, selection=
                        'ZZsel > 90 && ' + m_range[0] + '< ZZMass && ZZMass < ' + m_range[1])
            weights = tree2array(tree, branches='overallEventWeight', selection=
                        'ZZsel > 90 && ' + m_range[0] + '< ZZMass && ZZMass < ' + m_range[1])

            data_set, final_states, mask = post_selection_processing(data_set, features_specs[features_mode])

            if features_mode == 0:
                np.savetxt(cst.dir_suff_dict[0][0] + background + '_masks.ma', mask)
            else:
                mask = np.loadtxt(cst.dir_suff_dict[0][0] + background + '_masks.ma').astype(bool)

            data_set = data_set[mask]
            weights = weights[mask]
            final_states = final_states[mask]

            weights *= cst.cross_sections[background] / cst.event_numbers[background]

            nb_events = np.ma.size(data_set, 0)

            np.savetxt(directory + background + '_training.dst', data_set[:nb_events // 2])
            np.savetxt(directory + background + '_weights_training.wgt', weights[:nb_events // 2])
            np.savetxt(directory + background + '_finalstates_training.dst', final_states[:nb_events // 2])
            np.savetxt(directory + background + '_test.dst', data_set[nb_events // 2:])
            np.savetxt(directory + background + '_weights_test.wgt', weights[nb_events // 2:])
            np.savetxt(directory + background + '_finalstates_test.dst', final_states[nb_events // 2:])
            logging.info(background + ' weights, training and test sets successfully stored in ' + directory)


def identify_final_state(Z1_flavarr, Z2_flavarr):
    assert np.ma.size(Z1_flavarr) == np.ma.size(Z2_flavarr) 
    assert np.ma.size(Z1_flavarr) > 0
    tmp =  np.ones(len(Z1_flavarr)) # 2e2mu as a basis
    tmp[np.logical_and(Z1_flavarr == Z2_flavarr, Z1_flavarr == -121)] = 0 # 4 electrons
    tmp[np.logical_and(Z1_flavarr == Z2_flavarr, Z1_flavarr != -121)] = 2  # 4 muons
    return tmp


def read_root_files(modes, m_range=('118', '130')):
    '''
    Reads the root files for all production modes defined in constants, and outputs a first set of files 
    that still need to be merged, scaled, etc...
    :param modes: the features modes to be used (list of integers, should always start by 0 the _full features set)
    :param modes: the mass range in which events should be considered.
    '''
    for features_mode in modes:
        directory, suffix = cst.dir_suff_dict[features_mode]

        if os.path.isdir(directory):
            rmtree(directory)
        os.makedirs(directory)
        logging.info('Directory ' + directory + ' created')

        to_retrieve, dont_care1, dont_care2 = features_specs[features_mode]

        for prod_mode in cst.production_modes:
            rfile = r.TFile(base_path + prod_mode + '125/ZZ4lAnalysis.root')
            tree = rfile.Get('ZZTree/candTree')

            if prod_mode not in ['WminusH', 'WplusH', 'ZH']:
                data_set = tree2array(tree, branches=to_retrieve, selection=
                            'ZZsel > 90 && ' + m_range[0] + '< ZZMass && ZZMass < ' + m_range[1])
                weights = tree2array(tree, branches='overallEventWeight', selection=
                            'ZZsel > 90 && ' + m_range[0] + '< ZZMass && ZZMass < ' + m_range[1])

                data_set, final_states, mask = post_selection_processing(data_set, features_specs[features_mode])

                if features_mode == 0:
                    np.savetxt(cst.dir_suff_dict[0][0] + prod_mode + '_masks.ma', mask)
                else:
                    mask = np.loadtxt(cst.dir_suff_dict[0][0] + prod_mode + '_masks.ma').astype(bool)

                data_set = data_set[mask]
                weights = weights[mask]
                final_states = final_states[mask]

                nb_events = np.ma.size(data_set, 0)

                np.savetxt(directory + prod_mode + '_training.txt', data_set[:nb_events // 2])
                np.savetxt(directory + prod_mode + '_test.txt', data_set[nb_events // 2:])
                np.savetxt(directory + prod_mode + '_weights_training.txt', weights[:nb_events // 2])
                np.savetxt(directory + prod_mode + '_weights_test.txt', weights[nb_events // 2:])
                np.savetxt(directory + prod_mode + '_finalstates_training.txt', final_states[:nb_events // 2])
                np.savetxt(directory + prod_mode + '_finalstates_test.txt', final_states[nb_events // 2:])
                logging.info(prod_mode + ' weights, training and test sets successfully stored in ' + directory)

            elif prod_mode == 'ZH':
                decay_criteria = {'_lept': ' && genExtInfo > 10 && !(genExtInfo == 12 || genExtInfo == 14 || genExtInfo == 16)',
                                  '_hadr': ' && genExtInfo < 10',
                                  '_met': ' && (genExtInfo == 12 || genExtInfo == 14 || genExtInfo == 16)'}

                for decay in decay_criteria.keys():
                    data_set = tree2array(tree, branches=to_retrieve, selection=
                            'ZZsel > 90 && ' + m_range[0] + '< ZZMass && ZZMass < ' + m_range[1] + decay_criteria[decay])
                    weights = tree2array(tree, branches='overallEventWeight', selection=
                            'ZZsel > 90 && ' + m_range[0] + '< ZZMass && ZZMass < ' + m_range[1] + decay_criteria[decay])

                    data_set, final_states, mask = post_selection_processing(data_set, features_specs[features_mode])

                    if features_mode == 0:
                        np.savetxt(cst.dir_suff_dict[0][0] + prod_mode + decay + '_masks.ma', mask)
                    else:
                        mask = np.loadtxt(cst.dir_suff_dict[0][0] + prod_mode + decay + '_masks.ma').astype(bool)

                    data_set = data_set[mask]
                    weights = weights[mask]
                    final_states = final_states[mask]
                    nb_events = np.ma.size(data_set, 0)

                    np.savetxt(directory + prod_mode + decay + '_training.txt', data_set[:nb_events // 2])
                    np.savetxt(directory + prod_mode + decay + '_test.txt', data_set[nb_events // 2:])
                    np.savetxt(directory + prod_mode + decay + '_weights_training.txt', weights[:nb_events // 2])
                    np.savetxt(directory + prod_mode + decay + '_weights_test.txt', weights[nb_events // 2:])
                    logging.info(prod_mode + decay + ' weights, training and test sets stored in ' + directory)
                    np.savetxt(directory + prod_mode + decay + '_finalstates_training.txt', final_states[:nb_events // 2])
                    np.savetxt(directory + prod_mode + decay + '_finalstates_test.txt', final_states[nb_events // 2:])

            else:
                decay_criteria = {'_lept': ' && genExtInfo > 10', '_hadr': ' && genExtInfo < 10'}

                for decay in decay_criteria.keys():
                    data_set = tree2array(tree, branches=to_retrieve, selection=
                            'ZZsel > 90 && ' + m_range[0] + '< ZZMass && ZZMass < ' + m_range[1] + decay_criteria[decay])
                    weights = tree2array(tree, branches='overallEventWeight', selection=
                            'ZZsel > 90 && ' + m_range[0] + '< ZZMass && ZZMass < ' + m_range[1] + decay_criteria[decay])

                    data_set, final_states, mask = post_selection_processing(data_set, features_specs[features_mode])
                    if features_mode == 0:
                        np.savetxt(cst.dir_suff_dict[0][0] + prod_mode + decay + '_masks.ma', mask)
                    else:
                        mask = np.loadtxt(cst.dir_suff_dict[0][0] + prod_mode + decay + '_masks.ma').astype(bool)

                    data_set = data_set[mask]
                    weights = weights[mask]
                    final_states = final_states[mask]
                    nb_events = np.ma.size(data_set, 0)

                    np.savetxt(directory + prod_mode + decay + '_training.txt', data_set[:nb_events // 2])
                    np.savetxt(directory + prod_mode + decay + '_test.txt', data_set[nb_events // 2:])
                    np.savetxt(directory + prod_mode + decay + '_weights_training.txt', weights[:nb_events // 2])
                    np.savetxt(directory + prod_mode + decay + '_weights_test.txt', weights[nb_events // 2:])
                    np.savetxt(directory + prod_mode + decay + '_finalstates_training.txt', final_states[:nb_events // 2])
                    np.savetxt(directory + prod_mode + decay + '_finalstates_test.txt', final_states[nb_events // 2:])
                    logging.info(prod_mode + decay + ' weights, training and test sets stored in ' + directory)



def merge_vector_modes(modes=(0, 1)):
    for mode in modes:
        directory, no_care = cst.dir_suff_dict[mode]
        for decay in ['_lept', '_hadr']:
            file_list = [directory + mediator + decay for mediator in ['WplusH', 'WminusH', 'ZH']]

            training_set = np.loadtxt(file_list[0] + '_training.txt')
            test_set = np.loadtxt(file_list[0] + '_test.txt')
            weights_train = np.loadtxt(file_list[0] + '_weights_training.txt')
            weights_test = np.loadtxt(file_list[0] + '_weights_test.txt')
            finalstates_train = np.loadtxt(file_list[0] + '_finalstates_training.txt')
            finalstates_test = np.loadtxt(file_list[0] + '_finalstates_test.txt')

            # Rescale the events weights to have common cross_sections & event numbers equal to the ones of WplusH
            for idx, filename in enumerate(file_list[1:]):
                temp_train = np.loadtxt(filename + '_training.txt')
                temp_test = np.loadtxt(filename + '_test.txt')
                temp_weights_train = np.loadtxt(filename + '_weights_training.txt')
                temp_weights_test = np.loadtxt(filename + '_weights_test.txt')
                temp_finalstates_train = np.loadtxt(filename + '_finalstates_training.txt')
                temp_finalstates_test = np.loadtxt(filename + '_finalstates_test.txt')

                temp_weights_train *= cst.event_numbers['WplusH'] / cst.event_numbers[filename.split('/')[-1].split('_')[0]]
                temp_weights_test *= cst.event_numbers['WplusH'] / cst.event_numbers[filename.split('/')[-1].split('_')[0]]
                temp_weights_train *= cst.cross_sections[filename.split('/')[-1].split('_')[0]] / cst.cross_sections['WplusH']
                temp_weights_test *= cst.cross_sections[filename.split('/')[-1].split('_')[0]] / cst.cross_sections['WplusH']

                training_set = np.concatenate((training_set, temp_train), axis=0)
                test_set = np.concatenate((test_set, temp_test), axis=0)
                weights_train = np.concatenate((weights_train, temp_weights_train), axis=0)
                weights_test = np.concatenate((weights_test, temp_weights_test), axis=0)
                finalstates_train = np.concatenate((finalstates_train, temp_finalstates_train), axis=0)
                finalstates_test = np.concatenate((finalstates_test, temp_finalstates_test), axis=0)

            np.savetxt(directory + 'VH' + decay + '_training.txt', training_set)
            np.savetxt(directory + 'VH' + decay + '_test.txt', test_set)
            np.savetxt(directory + 'VH' + decay + '_weights_training.txt', weights_train)
            np.savetxt(directory + 'VH' + decay + '_weights_test.txt', weights_test)
            np.savetxt(directory + 'VH' + decay + '_finalstates_training.txt', finalstates_train)
            np.savetxt(directory + 'VH' + decay + '_finalstates_test.txt', finalstates_test)
    logging.info('Merged data successfully generated')


def make_full_datasets(modes=(0, 1, 2)):
    for mode in modes:
        directory, no_care = cst.dir_suff_dict[mode]

        file_list = [directory + cat for cat in cst.event_categories]
        training_set = np.loadtxt(file_list[0] + '_training.txt')
        test_set = np.loadtxt(file_list[0] + '_test.txt')
        training_labels = np.zeros(np.ma.size(training_set, 0))
        test_labels = np.zeros(np.ma.size(test_set, 0))
        training_weights = np.loadtxt(file_list[0] + '_weights_training.txt') * \
                           cst.cross_sections[cst.event_categories[0]] / cst.event_numbers[cst.event_categories[0]]
        test_weights = np.loadtxt(file_list[0] + '_weights_test.txt') * \
                       cst.cross_sections[cst.event_categories[0]] / cst.event_numbers[cst.event_categories[0]]

        training_finalstates = np.loadtxt(file_list[0] + '_finalstates_training.txt')
        test_finalstates = np.loadtxt(file_list[0] + '_finalstates_test.txt')


        for idx, filename in enumerate(file_list[1:]):
            temp_train = np.loadtxt(filename + '_training.txt')
            temp_test = np.loadtxt(filename + '_test.txt')
            tmp_training_weights = np.loadtxt(filename + '_weights_training.txt') * \
                                   cst.cross_sections[filename.split('/')[-1]] / cst.event_numbers[filename.split('/')[-1]]
            tmp_test_weights = np.loadtxt(filename + '_weights_test.txt') * \
                               cst.cross_sections[filename.split('/')[-1]] / cst.event_numbers[filename.split('/')[-1]]
            tmp_training_finalstates = np.loadtxt(filename + '_finalstates_training.txt')
            tmp_test_finalstates = np.loadtxt(filename + '_finalstates_test.txt')

            training_set = np.concatenate((training_set, temp_train), axis=0)
            test_set = np.concatenate((test_set, temp_test), axis=0)
            training_labels = np.concatenate((training_labels, (idx + 1) * np.ones(np.ma.size(temp_train, 0))), axis=0)
            test_labels = np.concatenate((test_labels, (idx + 1) * np.ones(np.ma.size(temp_test, 0))), axis=0)
            training_weights = np.concatenate((training_weights, tmp_training_weights), axis=0)
            test_weights = np.concatenate((test_weights, tmp_test_weights), axis=0)
            test_finalstates = np.concatenate((test_finalstates, tmp_test_finalstates), axis=0)
            training_finalstates = np.concatenate((training_finalstates, tmp_training_finalstates), axis=0)

        np.savetxt(directory + 'full_training_set.dst', training_set)
        np.savetxt(directory + 'full_training_labels.lbl', training_labels)
        np.savetxt(directory + 'full_training_weights.wgt', training_weights)
        np.savetxt(directory + 'full_test_set.dst', test_set)
        np.savetxt(directory + 'full_test_labels.lbl', test_labels)
        np.savetxt(directory + 'full_test_weights.wgt', test_weights)
        np.savetxt(directory + 'full_test_finalstates.dst', test_finalstates)
        np.savetxt(directory + 'full_training_finalstates.dst', training_finalstates)


def clean_intermediate_files(modes=(0, 1, 2)):
    for mode in modes:
        directory, no_care = cst.dir_suff_dict[mode]
        files_list = os.listdir(directory)
        for file_name in files_list:
            if file_name.split('.')[-1] not in ['dst', 'pkl', 'lbl', 'wgt']:
                os.remove(directory + file_name)


def full_process(modes=tuple(range(2)), m_range=('118', '130')):
    logging.info('Reading root files')
    read_root_files(modes, m_range)
    logging.info('Merging vector modes')
    merge_vector_modes(modes)
    logging.info('Merging and scaling datasets')
    make_full_datasets(modes)
    logging.info('Getting background files')
    get_background_files(modes, m_range)
    logging.info('Removing all intermediate files')
    clean_intermediate_files(modes)


def get_count(mode, idx=40):
    rfile = r.TFile(base_path + mode + '/ZZ4lAnalysis.root')
    counter = rfile.Get('ZZTree/Counters')
    plop = counter[idx]
    print(plop)


