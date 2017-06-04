import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.cm as cm
import matplotlib.pyplot as p
from core.evaluation import content_plot
import os
import logging
decision_stump = DecisionTreeClassifier(max_depth=1)



def custom_roc():
    extension = '_nomass'
    available_models = ['_'.join(full_name.split('_')[:-1]) for full_name in os.listdir('saves/metrics/') if
                        full_name.split('_')[-1] == 'acceptance.txt']
    stumps_dict = [name for name in available_models if '_'.join(name.split('_')[0:2]) == 'adaboost_stumps']
    slow_stumps_dict = [name for name in available_models if '_'.join(name.split('_')[0:2]) == 'adaslow03_stumps']
    p.figure()
    p.xlim(0., 1.)
    p.ylim(0., 1.)
    p.xlabel('Specificity')
    p.ylabel('Acceptance')
    p.title('Specifity vs Acceptance in VBF category with featureset ' + extension + '\n')

    acceptance = 0.47
    specificity = 0.37
    p.scatter(specificity, acceptance, marker='*', c='g', s=10**2, label='Legacy Mor17 2j only')
    acceptance = 0.73
    specificity = 0.15
    p.scatter(specificity, acceptance, marker='o', c='gr', s=10**2, label='Legacy Mor17 1j and 2j')

    for n_est, symbol in zip([300, 500, 1000], ['o', 'v', '^']):
        plop = [model for model in slow_stumps_dict if (model.split('_')[2] == str(n_est))]
        plop.sort()

        acceptances = np.array([np.loadtxt('saves/metrics/' + name + '_acceptance.txt')[1] for name in plop])
        specificities = np.array([np.loadtxt('saves/metrics/' + name + '_specificity.txt')[1] for name in plop])
        #mask = acceptances > 0.12
	#specificities = specificities[mask]
        #acceptances = acceptances[mask]

        p.scatter(specificities, acceptances, marker=symbol, c='b', label=str(n_est) + 'slower stumps')

    for n_est, symbol in zip([100, 200, 300, 500], ['o', 'v', '^', 'x']):
        plop = [model for model in stumps_dict if (model.split('_')[2] == str(n_est))]
        plop.sort()
        acceptances = [np.loadtxt('saves/metrics/' + name + '_acceptance.txt')[1] for name in plop]
        specificities = [np.loadtxt('saves/metrics/' + name + '_specificity.txt')[1] for name in plop]
        #mask = acceptances > 0.12
	#specificities = specificities[mask]
        #acceptances = acceptances[mask]
        p.scatter(specificities, acceptances, marker=symbol, c='r', label=str(n_est) + ' stumps')

    p.legend(loc=1)
    p.savefig('saves/figs/full_roc')
    p.show()


def check_weight_influence():
    extension = '_nomass'
    available_models = ['_'.join(full_name.split('_')[:-1]) for full_name in os.listdir('saves/metrics/') if
                        full_name.split('_')[-1] == 'acceptance.txt']
    stumps_dict = [name for name in available_models if '_'.join(name.split('_')[0:2]) == 'adaboost_stumps']
    slow_stumps_dict = [name for name in available_models if '_'.join(name.split('_')[0:2]) == 'adaslow03_stumps']
    p.figure()
    p.xlim(0., 1.)
    p.ylim(0., 1.)
    p.xlabel('Specificity')
    p.ylabel('Acceptance')
    p.title('Specifity vs Acceptance in VBF category with featureset ' + extension + '\n')

    acceptance = 0.47
    specificity = 0.37
    p.scatter(specificity, acceptance, marker='*', c='g', s=10**2, label='Legacy Mor17 2j only')
    acceptance = 0.73
    specificity = 0.15
    p.scatter(specificity, acceptance, marker='o', c='gr', s=10**2, label='Legacy Mor17 1j and 2j')

    for n_est, symbol in zip([300], ['o']):
        plop = [model for model in slow_stumps_dict if (model.split('_')[2] == str(n_est))]
        plop.sort()
        acceptances = np.array([np.loadtxt('saves/metrics/' + name + '_acceptance.txt')[1] for name in plop])
        specificities = np.array([np.loadtxt('saves/metrics/' + name + '_specificity.txt')[1] for name in plop])
        p.scatter(specificities, acceptances, marker=symbol, c=range(len(acceptances)) ,cmap=cm.autumn, label=str(n_est) + ' slower stumps')

    p.legend(loc=1)
    p.savefig('saves/figs/weight_influence')
    p.show()



def find_best_model():
    models = np.array(['_'.join(full_name.split('_')[:-1]) for full_name in os.listdir('saves/metrics/') if
                        full_name.split('_')[-1] == 'acceptance.txt'])

    for cat in range(1, 2):
        # Could be improved to allow for exploration of other regions
        acceptances = np.array([np.loadtxt('saves/metrics/' + name + '_acceptance.txt')[cat] for name in models])
        specificities = np.array([np.loadtxt('saves/metrics/' + name + '_specificity.txt')[cat] for name in models])

        mask = specificities > 0.47    # consider only models at least as pure as Legacy in VBF
        models = models[mask]
        acceptances = acceptances[mask]
        specificities = specificities[mask]

        max_acceptance = np.max(acceptances)
        mask_2 = acceptances > 0.97 * max_acceptance
        models = models[mask_2]
    #print(models) 
    for model in models :
        content_plot('_'.join(model.split('_')[:-1]), save=True)



def main():
    custom_roc()
    #check_weight_influence()
    #find_best_model()


if __name__ == "__main__":
   main()

