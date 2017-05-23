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

def add_logreg(base_dict = {}):
    for n_est in [50, 100, 200]:
        for purity_param in np.arange(10, 70, step=10):
            base_dict['adaboost_logreg_' + str(n_est) + '_' + str(purity_param) + '_' + 'custom'] = \
            (AdaBoostClassifier(LogisticRegression(solver='newton-cg', multi_class='ovr', n_jobs=8),
                                n_estimators=n_est), [float(purity_param) / 10., 1., 1., 1., 1., 1., 1.])


def custom_roc():
    extension = '_nomass'
    stumps_dict = add_stumps().keys()
    logreg_dict = add_logreg().keys()


    p.figure()
    p.xlim(0, 1)
    p.ylim(0, 1)
    p.xlabel('Purity')
    p.ylabel('Acceptance')
    p.title('Purity vs Acceptance plot for all studied models and set ' + extension)

    for n_est, symbol in zip([300, 500, 1000], ['o', 'v', '^']):
        plop = [model for model in stumps_dict if (model.split('_')[2] == n_est)]
        plop.sort()
        purities = [np.mean(np.loadtxt('saves/metrics/' + name + extension + '_purity.txt')) for name in plop]
        acceptances = [np.mean(np.loadtxt('saves/metrics/' + name + extension + '_acceptance.txt')) for name in plop]
        p.scatter(purities, acceptances, marker=symbol, c=range(len(plop)), cmap=cm.autumn ,
                  label=str(n_est) + ' stumps')

    for n_est, symbol in zip([50, 100, 200], ['o', 'v', '^']):
        plop = [model for model in logreg_dict if (model.split('_')[2] == n_est and model.split('_')[-2] == 'extension')]
        plop.sort()
        purities = [np.mean(np.loadtxt('saves/metrics/' + name + extension + '_purity.txt')) for name in plop]
        acceptances = [np.mean(np.loadtxt('saves/metrics/' + name + extension + '_acceptance.txt')) for name in plop]
        p.scatter(purities, acceptances, marker=symbol, c=range(len(plop)), cmap=cm.winter ,
                  label=str(n_est) + ' logregs')

    p.legend()
    p.show()




