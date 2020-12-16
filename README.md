# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
This dataset is related with a direct marketing campaigns of a Portuguese bank (1). We seek to predict if a customer will subscribe 
a term deposit. Knowing this information saves the banking company a lot of time and money because only customers with a high likelihood
of subscribing to the bank product are contacted via phone and informed about the term deposit. (1) 
We will compare a hyperparameter-tuned scikit-learn logistic regression model to a classification model created by Azure AutoML in this project.

The best performing model was a VotingEnsemble created by Azure AutoML which uses SparseNormalizer, MaxAbsScaler, StandardScaler as scalers and the classification algorithms XGBoost, LightGBM and logistic regression.

## Scikit-learn Pipeline

In this section I describe the pipeline architecture using the scikit-learn logistic regression classifier, which is displayed in the following flowchart.

![scikit-learn pipeline](./figures/Pipeline_MLProject1.png).

First the data is loaded from a csv file into TabularDataset. We use TabularDatasetFactory's from_delimited_files method for this task by specifying 
the URL where the csv file is located as a parameter of this function. After that the data set is cleaned using the clean_data function provided by Udacity. Now the dataset is splitted into a 
training set and a test set using the train_test_split function by scikit-learn. 

Before we could start hyperparameter tuning we have to set up a train.py script which is executed each hyperdrive run. In this script we load the data, clean and split it into train and test sets 
and train the scikit-learn logistic regression algorithm using parameters obtained from the arguments of the training script.
After the training process the performance of the classification on the test set is evaluated and appended to a log file. 

In the jupyter notebook where we manage the experiments, we create an SKLearn estimator by specifying the directory and name of the training script created in the last step and the compute target where the compution should be executed. 
Errors in earlier runs made it neccessary to specify pyarrow and pyspark as additional pip modules, so that these modules are added to the docker container created by AzureML when executing the pipeline.

The hyperdrive configuration consist of a hyperparameter sampling strategy  and an early stopping policy. The hyperparameter strategy defines how values for the hyperparameters are generated. The early stopping ploicy is used to stop runs
with a lower performance to save cost and time.

In my experiments I used RandomParameterSampling as hyperparameter sampling strategy, because it is computationally efficient and offers a good chance to find the optimum compared to a grid search. Another parameter sampler 
is BayesianParameterSampling, but this strategy does not support early stopping, because it uses former runs to improve the hyperparameter sampling results.

The next important parameter of the Hyperdrive configuration is the early stopping policy. I chose the BanditPolicy with a delay of three runs and a slack_factor of 1%. The policy is checked each run.
The Bandit Policy cancels a run if the difference of the specified metric of this run to the best run is higher than 1%. This leads to quicker computation times and lower expenses because runs with a smaller performance are stopped.

The last step is the submission  of the experiment by passing the hyperdrive config as an argument to the submit method of the experiment object. We could see the results using the 
RunDetails class. As you can see in the model output in the jupyter notebook the best hyperdrive run has an accuracy of 0.9163 and uses 0.85 as inverse regularization parameter C and sets the number of maximum iterations to 200.
The best model is saved using joblib and registered in Azure.

## AutoML
AutoML allows us to automatically train multiple models and their hyperparameters which are compared using the metric specified. To use AutoML an AutoML config has to be specified. In this configwe specify that 
SVMs should be blocked because it takes too much time to train them. 
The best model returned by AutoML is a Voting Ensemble method which means that multiple models are used to determine the class prediction for each instance. This practice is called ensemble method.
The TOP3 important features used by the VotingEnsemble model are the duration (duration of last call in seconds), the number of employees and the employment variation rate as you could see in the following plot.
![feature importance](./figures/feature_importance.png).
The classification algorithms and parameters used for the VotingEnsemble could be found in the html file in the repo ![Voting Ensemble](./udacity-project-2020-12-16.html).

## Pipeline comparison
The difference in accuracy between the best hyperparameter-tuned scikit-learn logistic regression model and the Voting Ensemble model from AutoML is very small (0.9163 vs. 0.9167). In my opinion this difference is neglectable and occurs due to randomness.
The major difference in architecture of the two models is that the logistic regression uses only one model to decide if a person will subscribe to the term deposit and the AutoML model uses multiple models for this purpose. Their predictions are aggregated using a majority vote principle.
 

## Future work
As you could see in the output of the AutoML run the dataset is imbalanced which influences the model performance negatively. To get rid of this influence you could under- or oversample the dataset using SMOTE or another sampling algorithm. 
By resampling the dataset the distribution of the target will be equal. The model performance increases because the model is trained on a dataset without a minority class.

Another area of improvement lies in the choice of the primary metric. Accuracy is calculated by dividing the correct classified instances by the number of total instances. Because of that 
imbalanced datasets lead to a high accuracy value. If you use a more robust metric like the area under curve (AUC), the area under the precision recall curve (AUPRC) or the f_1 metric you will get an 
unbiased information about the performance of your model.

The last point I want to elaborate on is the high correlation between the feature duration and the target variable class, which is referenced here https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
The high correlation could lead to a overestimation of model performance. In addition the duration of the contact is not known before the call, so that variable is not helpful when predicting the output 
on customers. Because of these reasons I would exclude this variable in further experiments.

## Resources
(1) Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.  (https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
