# Stage m2 : higgs generation mechanism classification


#### Using Monte-Carlo simulated data, train neural network classifiers, "pickle" them for later use, and generate content plots.  
## The whole program is controlled by the main.py script, calling the functions from the src package :
- src.constants regroups most of the program's global variables (the remaining are in preprocessing to regroup all ROOT commands in the same file)
- src.preprocessing generates .txt saves of the datasets with control over the features to retrieve / compute / remove.
- src.trainer trains the specified classifier and stores it in an appropriate folder along with its predictions.
- src.plotter takes a trained model, a test set, and generates the associated category content plot.

- In progress : SelfThresholdingAdaClassifier provides a wrapper of sklearn's adaboost meta-estimator designed to provide more control over its prediction method (especially optimize some prediction thresholds to minimize a given cost).

