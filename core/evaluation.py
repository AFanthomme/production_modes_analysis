import os
from copy import copy
from itertools import izip
import matplotlib.pyplot as p
import numpy as np
import core.trainer as ctg
import core.constants as cst
import cPickle as pickle
import pandas as pd
import matplotlib.cm as cm

def calculate_metrics(model_name):
    no_care, suffix = cst.dir_suff_dict[cst.features_set_selector]
    
    for model_name in [model_name + suffix, model_name + '_stacked' + suffix, model_name + '_doublestacked' + suffix]:
        suffix += '/'
    
        true_categories = np.loadtxt('saves/common' + suffix + 'full_test_labels.lbl')
        weights = np.loadtxt('saves/common' + suffix + 'full_test_weights.wgt')
        predictions = np.loadtxt('saves/predictions/' + model_name + '_predictions.prd')
    
        nb_categories = len(cst.event_categories)
        contents_table = np.zeros((nb_categories, nb_categories))
    
        for true_tag, predicted_tag, rescaled_weight in izip(true_categories, predictions, weights):
            contents_table[predicted_tag, true_tag] += rescaled_weight
    
        contents_table *= cst.luminosity
        correct_in_cat = [contents_table[cat, cat] for cat in range(nb_categories)]
        wrong_in_cat = np.sum(np.where(np.logical_not(np.identity(nb_categories, dtype=bool)), contents_table, 0), axis=1)
        cat_total_content = np.sum(contents_table, axis=0)
    
        bkg_predictions = np.loadtxt('saves/predictions/' + model_name + '_bkg_predictions.prd')
        bkg_weights = np.loadtxt('saves/common' + suffix + 'ZZTo4l_weights_test.wgt')
        bkg_repartition = np.array([np.sum(bkg_weights[np.where(bkg_predictions == cat)]) for cat in range(nb_categories)])
        bkg_repartition *= cst.luminosity
        specificity = [1. / (1. + (bkg_repartition[cat] + wrong_in_cat[cat]) / correct_in_cat[cat]) for cat in range(nb_categories)]
        acceptance = [correct_in_cat[cat] / cat_total_content[cat] for cat in range(nb_categories)]
        np.savetxt('saves/metrics/' + model_name + '_specificity.txt', specificity)
        np.savetxt('saves/metrics/' + model_name + '_acceptance.txt', acceptance)
        np.savetxt('saves/metrics/' + model_name + '_bkgrepartition.txt', bkg_repartition)
        np.savetxt('saves/metrics/' + model_name + '_contentstable.txt', contents_table)

def make_pretty_table(model_name):
    fs_labels = ['_4e', '_2e2mu', '_4mu']

    no_care, suffix = cst.dir_suff_dict[cst.features_set_selector]
    model_name += suffix
    suffix += '/'

    if not os.path.isdir('saves/tables_latex/'):
        os.makedirs('saves/tables_latex')

    true_categories_ref = np.loadtxt('saves/common' + suffix + 'full_test_labels.lbl')
    weights_ref = np.loadtxt('saves/common' + suffix + 'full_test_weights.wgt')
    predictions_ref = np.loadtxt('saves/predictions/' + model_name + '_predictions.prd')
    final_states = np.loadtxt('saves/common' + suffix + 'full_test_finalstates.dst').astype(int)
    bkg_predictions_ref = np.loadtxt('saves/predictions/' + model_name + '_bkg_predictions.prd')
    bkg_weights_ref = np.loadtxt('saves/common' + suffix + 'ZZTo4l_weights_test.wgt')
    bkg_final_states = np.loadtxt('saves/common' + suffix + 'ZZTo4l_finalstates_test.dst').astype(int)
    bkg_weights_ref *= 0.5 # Here there was no train/test split 
    nb_categories = len(cst.event_categories)
    nb_processes = nb_categories + 1   # Consider all background at once

    for fs_idx, fs_label in enumerate(fs_labels):
        contents_table = np.zeros((nb_categories, nb_processes))
        mask_fs = np.where(final_states == fs_idx)
        true_categories = true_categories_ref[mask_fs]
        weights = weights_ref[mask_fs]
        predictions = predictions_ref[mask_fs]
       
        bkg_mask_fs = np.where(bkg_final_states == fs_idx)
        bkg_weights = bkg_weights_ref[bkg_mask_fs]
        bkg_predictions = bkg_predictions_ref[bkg_mask_fs]

        for true_tag, predicted_tag, rescaled_weight in izip(true_categories, predictions, weights):
            contents_table[predicted_tag, true_tag] += rescaled_weight

        for predicted_tag, rescaled_weight in izip(bkg_predictions, bkg_weights):
            contents_table[predicted_tag, -1] += rescaled_weight
        
        contents_table *= cst.luminosity

        contents_table = np.around(contents_table, 5)
        row_labels = [cat + '_tagged' for cat in cst.event_categories]
        col_labels = cst.event_categories + ['ZZ4l']
        dataframe = pd.DataFrame(contents_table, index=row_labels, columns=col_labels)
        pretty_table = dataframe.to_latex()
        #print(pretty_table)
        with open('saves/tables_latex/' + model_name + fs_label, 'w') as f:
            f.write(pretty_table)

def content_plot(model_name, save=False):
    tags_list = copy(cst.event_categories)
    nb_categories = len(cst.event_categories)
    no_care, suffix = cst.dir_suff_dict[cst.features_set_selector]
    model_name += suffix
    suffix += '/'
   
    try:
        contents_table = np.loadtxt('saves/metrics/' + model_name + '_contentstable.txt')
        specificity = np.loadtxt('saves/metrics/' + model_name + '_specificity.txt')
        acceptance = np.loadtxt('saves/metrics/' + model_name + '_acceptance.txt')
        bkg_repartition = np.loadtxt('saves/metrics/' + model_name + '_bkgrepartition.txt')
    except:
        model_name = '_'.join((model_name.split('_')[:-2])) + '_nomass_stacked'
        contents_table = np.loadtxt('saves/metrics/' + model_name + '_contentstable.txt')
        specificity = np.loadtxt('saves/metrics/' + model_name + '_specificity.txt')
        acceptance = np.loadtxt('saves/metrics/' + model_name + '_acceptance.txt')
        bkg_repartition = np.loadtxt('saves/metrics/' + model_name + '_bkgrepartition.txt')

    ordering = [nb_categories - 1 - i for i in range(nb_categories)]

    fig = p.figure()
    p.title('Content plot for ' + model_name, y=-0.12)
    ax = fig.add_subplot(111)
    color_array = ['b', 'g', 'r', 'brown', 'm', '0.75', 'c', 'b']

    for category in range(nb_categories):
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
        ax.text(0.01, (position + 0.5) * 0.19 - 0.025, tags_list[category] + ', ' +
                str(np.round(np.sum(contents_table[category, :]), 2)) + r' events; $\mathcal{S} = $' +
                str(np.round(specificity[category], 3)) + r'$; \mathcal{A} =$' + str(np.round(acceptance[category], 3))
                , fontsize=16, color='w')

    ax.get_yaxis().set_visible(False)
    p.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=6, fontsize=11, mode="expand", borderaxespad=0.)
    if save:
        fig.savefig('saves/figs/' + model_name + '_content_plot.png')
    else:
        p.show()
    p.close(fig) 

    fig2 = p.figure(2)
    p.title('Per-category fraction of background events for ' + model_name, y=-0.12)
    ax = fig2.add_subplot(111)
    for category in range(nb_categories):
        position = ordering[category]
        ax.axhspan(position * 0.19 + 0.025, (position + 1) * 0.19 - 0.025, 0., bkg_repartition[category] / np.sum(contents_table[category,:]),
                  color='0.75', label=cst.event_categories[category])
        ax.text(0.01, (position + 0.5) * 0.19 - 0.025, tags_list[category] + ', ' + 
                str(np.round(bkg_repartition[category], 2)) + r' background events'
                , fontsize=16, color='b')

    ax.get_yaxis().set_visible(False)
    p.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=6, fontsize=11, mode="expand", borderaxespad=0.)
    if save:
        p.savefig('saves/figs/' + model_name + '_contamination_plot.png')
    p.close(fig2)

def feature_importance_plot(model_name):
    directory, suffix = cst.dir_suff_dict[cst.features_set_selector]
    with open('saves/classifiers/' + model_name + suffix + '_categorizer.pkl', 'rb') as f:
        classifier = pickle.load(f)
    if cst.features_set_selector == 2:
        features_names_xgdb = np.append(cst.features_names_xgdb, ['Z1_Flav', 'Z2_Flav'])
    else:  
        features_names_xgdb = cst.features_names_xgdb
        features_names_xgdb[2] = 'nb_jets'
        features_names_xgdb[3] = 'nb_b_jets'
        
    feat_imp = pd.Series(classifier.feature_importances_).sort_values(ascending=False)
    ordering = np.argsort(-classifier.feature_importances_)
    ax = feat_imp.plot(kind='bar', title='Features relative importance')
    ax.set_xticklabels(features_names_xgdb[ordering], rotation=30, fontsize=8)
    p.savefig('saves/figs/feature_importance.png')


def custom_roc():
    available_models = ['_'.join(full_name.split('_')[:-1]) for full_name in os.listdir('saves/metrics/') if
                        full_name.split('_')[-1] == 'acceptance.txt']

    p.figure()
    p.xlim(0., 1.)
    p.ylim(0., 1.)
    p.xlabel('Specificity')
    p.ylabel('Acceptance')
    p.title('Specifity vs Acceptance in the VBF category' + '\n')

    # These are PAS numbers using the 118-130 GeV, if we use the wider range the specificity should drop seriously.
    # Especially, if the models were trained on wider range, comparison unfair to them.
    acceptance = 0.47
    specificity = 0.37
    p.scatter(specificity, acceptance, marker='*', c='g', s=10 ** 2, label='Legacy Mor17 2j only')
    acceptance = 0.73
    specificity = 0.15
    p.scatter(specificity, acceptance, marker='o', c='gr', s=10 ** 2, label='Legacy Mor17 1j and 2j')

    plop = [model for model in available_models if (model.split('_')[0] == 'xgbslow')]
    plop.sort()
    acceptances = np.array([np.loadtxt('saves/metrics/' + name + '_acceptance.txt')[1] for name in plop])
    specificities = np.array([np.loadtxt('saves/metrics/' + name + '_specificity.txt')[1] for name in plop])
    coefs = np.polyfit(specificities, acceptances, 3)
    pol = np.poly1d(coefs)
    fit_range = np.linspace(np.min(specificities), np.max(specificities), 1024)
    p.scatter(specificities, acceptances, marker='o', c='r', label='xGDB trees')
    p.plot(fit_range, pol(fit_range), c='r')
    p.legend(loc=1)
    p.savefig('saves/figs/full_roc')
    p.show()


def check_weight_influence():
    available_models = ['_'.join(full_name.split('_')[:-1]) for full_name in os.listdir('saves/metrics/') if
                        full_name.split('_')[-1] == 'acceptance.txt']

    p.figure()
    p.xlim(0., 1.)
    p.ylim(0., 1.)
    p.xlabel('Specificity')
    p.ylabel('Acceptance')
    p.title('Evolution of the Specifity vs Acceptance in VBF categoryas a function of the rejection parameter \n')

    plop = [model for model in available_models if (model.split('_')[0] == 'xgbslow')]
    plop.sort()
    acceptances = np.array([np.loadtxt('saves/metrics/' + name + '_acceptance.txt')[1] for name in plop])
    specificities = np.array([np.loadtxt('saves/metrics/' + name + '_specificity.txt')[1] for name in plop])
    p.scatter(specificities, acceptances, marker='o', c=range(len(acceptances)), cmap=cm.autumn, label='xGDB trees')
    p.legend(loc=1)
    p.savefig('saves/figs/weight_influence')
    p.show()
