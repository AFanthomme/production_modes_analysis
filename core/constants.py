"""
Define all constants needed for what we want to do, and the sklearn models to use
"""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
#from core.custom_classifiers import SelfThresholdingAdaClassifier
import numpy as np
import pickle
import datetime

global_verbosity = 0
ignore_warnings = True

# Select one of the already defined modes. The selector is evaluated whenever it is needed, so that this value
# can easily be overriden from main.py
# To add new sets of features (either from file or calculated), add the corresponding file and suffix here then
# modify preprocessing.py
features_set_selector = 6

dir_suff_dict = [('saves/common_full/', '_full'), ('saves/common_onlydiscr/', '_onlydiscr'),
                 ('saves/common_nodiscr/', '_nodiscr'), ('saves/common_nomass/', '_nomass'),
                 ('saves/common_nomela/', '_nomela'), ('saves/common_nothing/', '_nothing'),
                 ('saves/common_linnomass/', '_linnomass')
                ]

production_modes = ['ggH', 'VBFH', 'WminusH', 'WplusH', 'ZH', 'ttH', 'bbH']
event_categories = ['ggH', 'VBFH', 'VH_hadr', 'VH_lept','ZH_met', 'ttH', 'bbH']


# These are the physical constants
luminosity = 2 * 35.9   # (fb-1), factor 2 because only half of the initial data set used for evaluation
cross_sections = {'ggH': 13.41, 'VBFH': 1.044, 'WminusH': 0.147, 'WplusH': 0.232, 'ZH': 0.668, 'ttH': 0.393,
                  'VH': 0.232, 'VH_lept': 0.232, 'VH_hadr': 0.232, 'bbH': 0.1347, 'ZH_met': 0.668,
                  'ZZTo4l': 1.256}
event_numbers = {'ZH': 376657.21875, 'WplusH': 252870.65625, 'WminusH': 168069.609375, 'ttH': 327699.28125,
                 'ggH': 999738.125, 'VBFH': 1885726.125, 'VH': 252870.65625, 'VH_lept': 252870.65625,
                 'VH_hadr': 252870.65625, 'bbH':327699.28125, 'ZH_met': 376657.21875, 'ZZTo4l': 6670241.5}




base_features = [
                'nExtraLep', 'nExtraZ', 'nCleanedJetsPt30', 'nCleanedJetsPt30BTagged_bTagSF',
                'p_JJQCD_SIG_ghg2_1_JHUGen_JECNominal', 'p_JQCD_SIG_ghg2_1_JHUGen_JECNominal',
                'p_JJVBF_SIG_ghv1_1_JHUGen_JECNominal', 'p_JVBF_SIG_ghv1_1_JHUGen_JECNominal',
                'pAux_JVBF_SIG_ghv1_1_JHUGen_JECNominal', 'p_HadWH_SIG_ghw1_1_JHUGen_JECNominal',
                'p_HadZH_SIG_ghz1_1_JHUGen_JECNominal', 'ZZMass', 'PFMET'
                ]

likelihood_names = ['p_JJQCD_SIG_ghg2_1_JHUGen_JECNominal', 'p_JQCD_SIG_ghg2_1_JHUGen_JECNominal',
                'p_JJVBF_SIG_ghv1_1_JHUGen_JECNominal', 'p_JVBF_SIG_ghv1_1_JHUGen_JECNominal',
                'pAux_JVBF_SIG_ghv1_1_JHUGen_JECNominal', 'p_HadWH_SIG_ghw1_1_JHUGen_JECNominal',
                'p_HadZH_SIG_ghz1_1_JHUGen_JECNominal']

backgrounds = ['ZZTo4l']

decision_stump = DecisionTreeClassifier(max_depth=1)

models_dict = {}

def add_stumps():
    for n_est in [300, 500, 1000]:
        for purity_param in np.arange(45, 100, step=5):
            models_dict['adaboost_stumps_' + str(n_est) + '_' + str(purity_param) + '_' + 'custom'] = \
            (AdaBoostClassifier(decision_stump, n_estimators=n_est), [float(purity_param) / 10., 1., 1., 1., 1., 1., 1.])

def add_logreg():
    for n_est in [50, 100, 200]:
        for purity_param in np.arange(10, 70, step=10):
            models_dict['adaboost_logreg_' + str(n_est) + '_' + str(purity_param) + '_' + 'custom'] = \
            (AdaBoostClassifier(LogisticRegression(solver='newton-cg', multi_class='ovr', n_jobs=8),
                                n_estimators=n_est), [float(purity_param) / 10., 1., 1., 1., 1., 1., 1.])

add_logreg()
add_stumps()
