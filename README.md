# Stage m2 : higgs events production modes classification


#### Using Monte-Carlo simulated data, train Machine Learning classifiers, quantify their performance (ROC curve, category content plot). 
## The program is controlled by the main.py script, calling the functions from the core package :
- constants regroups most of the program's global variables (the remaining are in preprocessing to regroup all ROOT commands in the same file)
- preprocessing generates .txt saves of the datasets with control over the features to retrieve / compute / remove.
- trainer trains the specified classifier and stores it in an appropriate folder along with its predictions.
- plotter takes a trained model, a test set, and generates the associated category content plot.
- roc_curve uses the metrics to generate our custom ROC plots.



## Most of the intermediate results are stored in the saves/ folder 

