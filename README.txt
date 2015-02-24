DATA FILES

census-just-names.txt: 
names and descriptions of variables in census data from UCI ML website

censusImputedNumeric.csv: 
final, cleaned training data

censusTestImputedNumeric.csv: 
final, cleaned test data

censusImputedNumeric_subalg.csv: 
experimental training data using random forest imputation

censusTestImputedNumeric_subalg.csv: experimental test data using random forest imputation


PYTHON FILES

chooseFeature.py: 
data preparation and cross validation for chooseFeature algorithm

KNN.py: 
data preparation, hyperparameter tuning and cross validation for knn algorithm

Logistic.py: 
data preparation, hyperparameter tuning and cross validation for logistic algorithm

MonicaGriffinLubaFinalProject.py: (MAIN CODE FILE FOR TRAINING AND TESTING!!)
all algorithms trained and applied to test data, produces confusion matrix for each model and prints scores

naiveBayes.py: 
data preparation and cross validation for naiveBayes algorithm

randomForest.py: 
data preparation, hyperparameter tuning and cross validation for randomForest algorithm

svm.py: 
data preparation, hyperparameter tuning and cross validation for SVM algorithm

tester.py:
imported this file in other files, used functions to transform data


R FILES

cleaning_and_eda.R: 
produces univariate plots, cleans and imputes test and training data

data_cleaning_subalg.R: 
file producing the experimental training and test data using random forest imputation

knn_logistic_hyperparam_vis.R: 
knn and logistic hyperparameter tuning visualization
