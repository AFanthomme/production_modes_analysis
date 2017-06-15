import numpy as np
from sklearn.tree import DecisionTreeClassifier
import matplotlib.cm as cm
import matplotlib.pyplot as p
from core.evaluation import content_plot
import os

decision_stump = DecisionTreeClassifier(max_depth=1)

def custom_roc():
    extension = '_nomass'
    available_models = ['_'.join(full_name.split('_')[:-1]) for full_name in os.listdir('saves/metrics/') if
                        full_name.split('_')[-1] == 'acceptance.txt']
    stumps_dict = [name for name in available_models if '_'.join(name.split('_')[0:2]) == 'adaboost_stumps']

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
    p.scatter(specificity, acceptance, marker='*', c='g', s=10**2, label='Legacy Mor17 2j only')
    acceptance = 0.73
    specificity = 0.15
    p.scatter(specificity, acceptance, marker='o', c='gr', s=10**2, label='Legacy Mor17 1j and 2j')
    
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
    extension = '_nomass'
    available_models = ['_'.join(full_name.split('_')[:-1]) for full_name in os.listdir('saves/metrics/') if
                        full_name.split('_')[-1] == 'acceptance.txt']
    stumps_dict = [name for name in available_models if '_'.join(name.split('_')[0:2]) == 'adaboost_stumps']
    p.figure()
    p.xlim(0., 1.)
    p.ylim(0., 1.)
    p.xlabel('Specificity')
    p.ylabel('Acceptance')
    p.title('Specifity vs Acceptance in VBF category with featureset ' + extension + '\n')


    # These are PAS numbers using the 118-130 GeV, if we use the wider range the specificity should drop seriously.
    # Especially, if the models were trained on wider range, comparison unfair to them.
    acceptance = 0.47
    specificity = 0.37
    p.scatter(specificity, acceptance, marker='*', c='g', s=10**2, label='Legacy Mor17 2j only')
    acceptance = 0.73
    specificity = 0.15
    p.scatter(specificity, acceptance, marker='o', c='gr', s=10**2, label='Legacy Mor17 1j and 2j')

    for n_est, symbol in zip([300], ['o']):
        plop = [model for model in stumps_dict if (model.split('_')[2] == str(n_est))]
        plop.sort()
        acceptances = np.array([np.loadtxt('saves/metrics/' + name + '_acceptance.txt')[1] for name in plop])
        specificities = np.array([np.loadtxt('saves/metrics/' + name + '_specificity.txt')[1] for name in plop])
        p.scatter(specificities, acceptances, marker=symbol, c=range(len(acceptances)) ,cmap=cm.autumn, label=str(n_est) + ' slower stumps')

    p.legend(loc=1)
    p.savefig('saves/figs/weight_influence')
    p.show()
