"""
Define all constants needed for what we want to do, and the sklearn models to use
"""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import warnings
global_verbosity = 0
ignore_warnings = True
import os
from xgboost.sklearn import XGBClassifier

os.environ['PATH'] = os.environ['PATH'] + ';C:\\Program Files\\mingw-w64\\x86_64-5.3.0-posix-seh-rt_v4-rev0\\mingw64\\bin'


features_set_selector = 1

if not os.path.isdir('saves/classifiers'):
    os.makedirs('saves/classifiers')

if not os.path.isdir('saves/figs'):
    os.makedirs('saves/figs')

if not os.path.isdir('saves/metrics'):
    os.makedirs('saves/metrics')

if not os.path.isdir('saves/predictions'):
    os.makedirs('saves/predictions')

if ignore_warnings:
    warnings.filterwarnings('ignore')

dir_suff_dict = [('saves/common_full/', '_full'), ('saves/common_nomass/', '_nomass'),
                 ('saves/common_flavors/', '_flavors')
                ]

production_modes = ['ggH', 'VBFH', 'WminusH', 'WplusH', 'ZH', 'ttH', 'bbH']

event_categories = ['ggH', 'VBFH', 'VH_hadr', 'VH_lept','ZH_met', 'ttH', 'bbH']

luminosity = 2 * 35.9   # factor 2 because only half of the initial data set used for evaluation

mass_range = ('105', '140')
cross_sections = {'ggH': 13.41, 'VBFH': 1.044, 'WminusH': 0.147, 'WplusH': 0.232, 'ZH': 0.668, 'ttH': 0.393,
                  'VH': 0.232, 'VH_lept': 0.232, 'VH_hadr': 0.232, 'bbH': 0.1347, 'ZH_met': 0.668,
                  'ZZTo4l': 1256.0}

event_numbers = {'ZH': 376657.21875, 'WplusH': 252870.65625, 'WminusH': 168069.609375, 'ttH': 327699.28125,
                 'ggH': 999738.125, 'VBFH': 1885726.125, 'VH': 252870.65625, 'VH_lept': 252870.65625,
                 'VH_hadr': 252870.65625, 'bbH':327699.28125, 'ZH_met': 376657.21875, 'ZZTo4l': 6670241.5}

features_names_xgdb = np.array(['nExtraLep', 'nExtraZ', 'nCleanedJetsPt30', 'nCleanedJetsPt30BTagged_bTagSF',
                  'PFMET', 'DVBF2j_ME', 'DVBF1j_ME', 'DWHh_ME', 'DZHh_ME'])


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

xgb_base = XGBClassifier(
     learning_rate =0.04,
     n_estimators=1000,
     max_depth=4,
     min_child_weight=4,
     gamma=0,
     subsample=0.8,
     colsample_bytree=0.8,
     objective= 'multi:softmax',
     num_class=7,
     n_jobs=16,
     )

decision_stump = DecisionTreeClassifier(max_depth=1)

models_dict = {'xgbslow_170': (xgb_base, [1.7, 1., 1., 1., 1., 1., 1.]),
               'xgbslow_500': (xgb_base, [5., 1., 1., 1., 1., 1., 1.]),
               }


def add_xgdb():
    for purity_param in np.arange(100, 500, step=10):
        models_dict['xgbslow' + '_' + str(purity_param)] = \
            (xgb_base, [float(purity_param) / 100., 1., 1., 1., 1., 1., 1.])

#add_xgdb()
