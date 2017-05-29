import numpy as np
import inspect
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
from core import constants as cst
# Common part of the path to retrieve the root files
base_path = '/data_CMS/cms/ochando/CJLSTReducedTree/170222/'

r.gROOT.LoadMacro("libs/cConstants_no_ext.cc")
r.gROOT.LoadMacro("libs/Discriminants_no_ext.cc")
r.gROOT.LoadMacro("libs/Category_no_ext.cc")
#print hasattr(r, 'categoryMor17')
inspect.getargspec(r.categoryMor17)
r.categoryMor17(1, 2, 3, 4, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, True)

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
                  'ZZTo4l': 1256.}
event_numbers = {'ZH': 376657.21875, 'WplusH': 252870.65625, 'WminusH': 168069.609375, 'ttH': 327699.28125,
                 'ggH': 999738.125, 'VBFH': 1885726.125, 'VH': 252870.65625, 'VH_lept': 252870.65625,
                 'VH_hadr': 252870.65625, 'bbH':327699.28125, 'ZH_met': 376657.21875, 'ZZTo4l': 6670241.5}
backgrounds = ['ZZTo4l']


def remove_fields(a, *fields_to_remove):
    return a[[name for name in a.dtype.names if name not in fields_to_remove]]

def read_root_files():
        directory = 'saves/legacy/'
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

        directory = 'saves/legacy/'

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
    directory = 'saves/legacy/'
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

def convert_types(a):
    ret = [None for _ in a]
    i= 0
    for x, newtype in zip(a, [int, int, int, int, float, float, float, float, float,  float, float, float, float, bool]):
       ret[i] = newtype(x)
       i += 1
    return ret


def generate_metrics():
    directory = 'saves/legacy/'
    if not os.path.isdir(directory):
        os.mkdir(directory)
    nb_categories = len(event_categories)
    contents_table = np.zeros((nb_categories+1, nb_categories))
#    reorder = [2, 3, 4, 6, 5, 1, 0]
    reorder = [0, 1, 2, 3, 4, 5, 6]
    for cat in range(nb_categories):
        plop = np.loadtxt('saves/legacy/' + event_categories[cat] + '_training.txt')
        plop_weights = np.loadtxt('saves/legacy/' + event_categories[cat] + '_weights_training.txt') * cst.cross_sections[event_categories[cat]] /event_numbers[event_categories[cat]]

        for event_idx in range(len(plop)):
            u = np.append(plop[event_idx, :], True)
            u = convert_types(u)
            nocool_identifier = r.categoryMor17(*u)
            contents_table[reorder[nocool_identifier], cat] += plop_weights[event_idx]
    contents_table *= luminosity

    ordering = [nb_categories - 1 - i for i in range(nb_categories+1)]

    fig = p.figure()
    p.title('Content plot for legacy categorization', y=-0.12)
    ax = fig.add_subplot(111)
    color_array = ['b', 'g', 'r', 'brown', 'm', '0.75', 'c', 'b']
    tags_list = range(nb_categories + 1)
    for category in range(nb_categories+1):
        position = ordering[category]
        normalized_content = contents_table[category, :].astype('float') / np.sum(contents_table[category, :])
        tmp = 0.
        for gen_mode in range(nb_categories):
            if position == 1:
                ax.axhspan(position * 0.19 + 0.025, (position + 1) * 0.19 - 0.025, tmp,
                           tmp + normalized_content[gen_mode],
                           color=color_array[gen_mode], label=cst.event_categories[gen_mode])
            else:
                ax.axhspan(position * 0.19 + 0.025, (position + 1) * 0.19 - 0.025, tmp,
                           tmp + normalized_content[gen_mode],
                           color=color_array[gen_mode])
            tmp += normalized_content[gen_mode]
        ax.text(0.01, (position + 0.5) * 0.19 - 0.025, str(tags_list[category]) + ', ' +
                str(np.round(np.sum(contents_table[category, :]), 2)) + r' events; $\mathcal{P} = $', fontsize=16, color='w')

    ax.get_yaxis().set_visible(False)
    p.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=6, fontsize=11, mode="expand", borderaxespad=0.)
    p.savefig('saves/figs/legacy_content_plot.png')

    bkg_weights = np.loadtxt('saves/legacy/' + 'ZZTo4l_weights.wgt')
    plop = np.loadtxt('saves/legacy/' + 'ZZTo4l.dst')
    bkg_repartition = np.zeros(nb_categories + 1)
    for event_idx in range(len(plop)):
        u = np.append(plop[event_idx, :], True)
        u = convert_types(u)
        nocool_identifier = r.categoryMor17(*u)
        bkg_repartition[nocool_identifier] += bkg_weights[event_idx]
    bkg_repartition *= cross_sections['ZZTo4l'] * 0.5 * cst.luminosity / cst.event_numbers['ZZTo4l']
    np.savetxt('saves/metrics/legacy' + '_bkgrepartition.txt', bkg_repartition)
    correct_in_cat = [0 for _ in range(nb_categories+1)]
    wrong_in_cat = [0 for _ in range(nb_categories+1)]
    
    real_cats = [0, 0, 1, 2, 3, 4, 5]
    for shitty_cat in range(nb_categories+1):
        correct_in_cat[shitty_cat] = contents_table[shitty_cat, real_cats[shitty_cat]]
        wrong_in_cat[shitty_cat] = np.sum(contents_table[shitty_cat, :]) - correct_in_cat[shitty_cat] 
   
    cat_total_content = np.sum(contents_table, axis=0)
    purity = [1. / (1. + (bkg_repartition[cat] + wrong_in_cat[cat]) / correct_in_cat[cat]) for cat in range(nb_categories+1)]
    acceptance = [correct_in_cat[cat] / cat_total_content[real_cats[cat]] for cat in range(nb_categories+1)]
    np.savetxt('saves/metrics/legacy_purity.txt', purity) 
    np.savetxt('saves/metrics/legacy_acceptance.txt', acceptance) 
    logging.info('Purity, acceptance : ' + str(np.mean(purity[1:])) + ', ' + str(np.mean(acceptance[1:])))

    fig = p.figure(2)
    p.title('Background contents for legacy classification', y=-0.12)
    ax = fig.add_subplot(111)
    for category in range(nb_categories):
        position = ordering[category]
        ax.axhspan(position * 0.19 + 0.025, (position + 1) * 0.19 - 0.025, 0., bkg_repartition[category] / np.sum(contents_table[category,:]),
                   color='0.75', label=cst.event_categories[category])
        ax.text(0.01, (position + 0.5) * 0.19 - 0.025, str(tags_list[category]) + ', ' +
                str(np.round(bkg_repartition[category], 2)) + r' bkg events', fontsize=16, color='b')

    ax.get_yaxis().set_visible(False)
    p.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=6, fontsize=11, mode="expand", borderaxespad=0.)
    p.savefig('saves/figs/legacy' + '_contamination_plot.png')


if __name__ == '__main__':
    read_root_files()
    get_background_files()
    merge_vector_modes()
    generate_metrics()
