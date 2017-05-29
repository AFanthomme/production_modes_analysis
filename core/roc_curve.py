import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.cm as cm
import matplotlib.pyplot as p

decision_stump = DecisionTreeClassifier(max_depth=1)



def add_stumps(base_dict = {}):
    for n_est in [300, 500, 1000]:
        for purity_param in np.arange(45, 100, step=5):
            base_dict['adaboost_stumps_' + str(n_est) + '_' + str(purity_param) + '_' + 'custom'] = \
            (AdaBoostClassifier(decision_stump, n_estimators=n_est), [float(purity_param) / 10., 1., 1., 1., 1., 1., 1.])
    return base_dict

def add_logreg(base_dict = {}):
    for n_est in [50, 100, 200]:
        for purity_param in np.arange(10, 70, step=10):
            base_dict['adaboost_logreg_' + str(n_est) + '_' + str(purity_param) + '_' + 'custom'] = \
            (AdaBoostClassifier(LogisticRegression(solver='newton-cg', multi_class='ovr', n_jobs=8),
                                n_estimators=n_est), [float(purity_param) / 10., 1., 1., 1., 1., 1., 1.])
    return base_dict

def add_stumps_slower(base_dict={}):
    for n_est in [300, 500, 1000]:
        for purity_param in np.arange(45, 100, step=5):
            base_dict['adaslow03_stumps_' + str(n_est) + '_' + str(purity_param) + '_' + 'custom'] = \
            (AdaBoostClassifier(decision_stump, n_estimators=n_est, learning_rate=0.3), [float(purity_param) / 10., 1., 1., 1., 1., 1., 1.])
    return base_dict

def custom_roc():
    extension = '_nomass'
    stumps_dict = add_stumps().keys()
    logreg_dict = add_logreg().keys()
    slow_stumps_dict = add_stumps_slower().keys()
 
    p.figure()
    p.xlim(0.2, 0.7)
    p.ylim(0.2, 0.7)
    p.xlabel('Purity')
    p.ylabel('Acceptance')
    p.title('Purity vs Acceptance plot for all studied models and set ' + extension)

    acceptance = np.loadtxt('saves/metrics/legacy_acceptance.txt')[1]
    puritie = np.loadtxt('saves/metrics/legacy_purity.txt')[1]
    #        purities = [np.mean(np.loadtxt('saves/metrics/' + name + extension + '_purity.txt')[1:]) for name in plop]
    #        acceptances = [np.mean(np.loadtxt('saves/metrics/' + name + extension + '_acceptance.txt')[1:]) for name in plop]
    p.scatter(puritie, acceptance, marker='*', c='b', s=4, label='Legacy Mor17')

    for n_est, symbol in zip([300, 500, 1000], ['o', 'v', '^']):
        plop = [model for model in slow_stumps_dict if (model.split('_')[2] == str(n_est))]
        plop.sort()
        acceptances = [np.loadtxt('saves/metrics/' + name + extension + '_acceptance.txt')[1] for name in plop]
        purities = [np.loadtxt('saves/metrics/' + name + extension + '_purity.txt')[1] for name in plop]
#        purities = [np.mean(np.loadtxt('saves/metrics/' + name + extension + '_purity.txt')[1:]) for name in plop]
#        acceptances = [np.mean(np.loadtxt('saves/metrics/' + name + extension + '_acceptance.txt')[1:]) for name in plop]
        p.scatter(purities, acceptances, marker=symbol, c='b', label=str(n_est) + 'slower stumps')

    for n_est, symbol in zip([300, 500, 1000], ['o', 'v', '^']):
        plop = [model for model in stumps_dict if (model.split('_')[2] == str(n_est))]
        plop.sort()
        acceptances = [np.loadtxt('saves/metrics/' + name + extension + '_acceptance.txt')[1] for name in plop]
        purities = [np.loadtxt('saves/metrics/' + name + extension + '_purity.txt')[1] for name in plop]
#        purities = [np.mean(np.loadtxt('saves/metrics/' + name + extension + '_purity.txt')[1:]) for name in plop]
#        acceptances = [np.mean(np.loadtxt('saves/metrics/' + name + extension + '_acceptance.txt')[1:]) for name in plop]
        p.scatter(purities, acceptances, marker=symbol, c='r', label=str(n_est) + ' stumps')
#
#    for n_est, symbol in zip([50, 100, 200], ['o', 'v', '^']):
#        plop = [model for model in logreg_dict if (model.split('_')[2] == str(n_est))]
#        plop.sort()
#        purities = [np.mean(np.loadtxt('saves/metrics/' + name + extension + '_purity.txt')) for name in plop]
#        acceptances = [np.mean(np.loadtxt('saves/metrics/' + name + extension + '_acceptance.txt')) for name in plop]
#        p.scatter(purities, acceptances, marker=symbol, c='g', label=str(n_est) + ' logregs')

    p.legend(loc=2)
    p.show()


if __name__ == "__main__":
    custom_roc()

