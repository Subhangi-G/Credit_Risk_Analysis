# Credit_Risk_Analysis

## Overview of Project 

### Purpose

The purpose of this project is to perform a predictive risk analysis for loan repayments on credit card clients, which is an inherently unbalanced classification problem because the low risk loans are much larger compared with risky loans.\
Credit risk was assessed in multiple ways. One apprach included using different algorithms that randomly resampled the training data to rebalance the class distribution before training a ML model. Another approach was to train and evaluate machine learning models that are designed to reduce bias in data with sevely skewed class distribution. 

### Resources used
- Data : LoanStats_2019Q1.csv, a credit card dataset from LendingClub
- Software : imbalanced_learn 0.8.0, scikit-learn 0.24.1, Numpy 1.20.1, Pandas 1.2.4

## Results and Analysis
The dataset used for this challenge has an inherently imbalanced outcome, and this has the potential to affect the performance of the trained model. To account for this imbalance, different approches were explored in this analysis.\
Algorithms that can deal with imbalanced data were employed. They use re-sampling techniques such as oversampling, undersampling and combinatorial approaches to recreate a balanced class distribution. The resampled data is then used to train a machine learning model such as logistic regression, which can then be used for prediction.\
Another way to work with such data is to use models that employ ensemble techniques, such as bagging, and boosting, which reduces bias produced by imbalanced class distribution.

Before applying any of the above approaches, the data was first loaded, and viewed as a DataFrame. There were 86 different columns, of which some were non-numeric.\
The non-numeric column, ***loan_status***, was determined to be the target variable (y) which contained two classes, low_risk and high_risk. There were 68,470 low_risk and 347 high_risk outcomes confirming the severity in skew.\
Every data point (rows) fell into one of the outcome classes. Every other feature (i.e. other than loan_status) would be part of the training data(X). The training variables with string values were converted to numerical values using dummy variables.\
The data was divided into the training and testing sets i.e. X_train, y_train, X-test, and y_test, and this was used in the various approaches to predict credit risk.

### Resampling algorithms to predict credit risk.
The training data was resampled using different types algorithms (explained below), and stored in X_resampled, and y_resampled. The resampled data was then used to train a logistic regression model.\
This model was then used to predict using the testing data, and the predicted outcome, y_pred was compared with the observed outcome, y_test. 
- A balanced accuracy score was calculated. 
- A confusion matrix was displayed.
- An imbalanced classification report was created.

Results are reported here.

#### Oversampling algorithms 

**RandomOverSampler**\
Oversampling was first performed using the RandomOverSampler algorithm. A picture of the result is included below.

![RandomOverSampler](https://user-images.githubusercontent.com/71800628/129619213-56763d83-fe92-41fe-bf22-5f7067bf94c1.png)


- The accuracy is around 67%.
- The precision of capturing defaulters is 1% meaning that out of all the high risk loans predicted by the model 1% are true defaulters. In contrast, 100% of the non-defaulters predicted by the model are truly low_risk (as precision for low risk was 100%)
- The recall (or sensitivity) for high risk is 70%. This means that 70% of the true risky loans were captured by the model, and 63% of the low risk loans were correctly predicted by the model (recall of low risk is 63%).

Notice that for high risk loans, precision is very low and recall is good. This gives a low F1 score, which is the harmonic mean of the precision and recall, and gives a measure of the balance between the two.\
Here we will focus on the precision and recall for high risk loans only because the goal is to be able to predict credit risk, therefore being able to predict the defaulters is more important than being able to predict non-defaulters.\
Also note that accuracy as a metric can be misleading when working with an imbalanced dataset (this is expained in detail in the summary section). 


**SMOTE**
Oversampling was performed using the SMOTE algorithm. A picture of the result is included below.

![SMOTE](https://user-images.githubusercontent.com/71800628/129619262-ed1e20af-6e55-4150-849c-eb21ca743f6f.png)


- The accuracy is 66%.
- The precision for high risk loans is 1%.
- The recall for high risk loans is 63%.

Over sampling using two different methods produced similar results, with RandomeOverSampler leading to slightly better recall than SMOTE.

#### Undersampling algorithm 
**ClusterCentroids**\
Undersampling was performed using the ClusterCentroid algorithm. A picture of the result is included below.

![ClusterCentroids](https://user-images.githubusercontent.com/71800628/129619286-365fca9d-bc22-4a34-8d73-362bc53439fe.png)


- The accuracy is around 54%.
- The precision for high risk loans is 1%. 
- The recall for high risk loans is 69%.

Notice that although the accuracy of the model is low compared to results obtained by oversampling, the recall for high risk is very similar to RandomOverSampling, and better than SMOTE.

#### Combinatorial approach algorithm
**SMOTEENN**\
A combinatorial approach of over-and-undersampling was undertaken using the SMOTEENN algorithm. A picture of the result is included below.

![SMOTEENN](https://user-images.githubusercontent.com/71800628/129619333-ffd89929-6f78-4d36-b08d-17d765218de7.png)


- The accuracy is around 64%.
- The precision for high risk loans is 1%. 
- The recall for high risk is loans 71%.

The combinatorial approch gave the best recall from all the resampling methods applied, although it wasn't a huge improvement over the other methods.\
The accuracy is similar too to the other methods (better than SMOTE). It is expected that accuracy will not be high because the data has been resampled to balance out the skew between the classes. 


### Ensemble Classifiers (models that reduce bias)
Here two separate models based on two separate methods, bagging and boosting, to reduce bias are used.\
The training data (X_train and y_train) is used to train the model. The model is then used for prediction using the testing set (X_test). The predicted y values, y_pred is then compared with the expected outcomes, y_test.\
As before the balanced accuracy score, confusion matrix, and a classification report is generated for each model.

#### Bagging method
**BalancedRandomForestClassifier**\
The BalancedRandomForestClassifier is an ensemble algorithm that uses subsets of the training data on multiple decision tree models, and then combines the predictions from all. The result is shown in the picture below.

![BalancedRandomForestClassifier](https://user-images.githubusercontent.com/71800628/129619369-4a9eb84c-bfdf-4014-90cf-81fdf8c88aa7.png)


- The accuracy is around 79%.
- The precision for high risk loans is 3%. 
- The recall for high risk loans is 70%.

Notice that accuracy is better than the results obtained when the training data was resampled before fitting to a logistic regression model. The recall is similar.

The input features, used for decision making in the random forest algorithm, is shown below in order of their relavance. The recall obtained by using the above method may be improved by employing feature selection using the results of the feature importance. (This part has not been implemented in this challenge). 

![BalancedRandomForestClassifier_FeatureImportances](https://user-images.githubusercontent.com/71800628/129619400-805d4509-fc71-478e-9cac-10a3afa0d0e7.png)


#### Boosting method
**EasyEnsembleClassifier**\
The EasyEnsembleClassifier is an AdaBoost algorithm that combines many weak classifier into a single strong classifier. The weak learner models are trained and applied sequantially to the ensemble. Each subsequent models weighs the training data to correct the prediction errors made by the previous model in the sequence. The result is shown in the picture below.

![EasyEnsambleClassifier](https://user-images.githubusercontent.com/71800628/129619423-40ab045a-0009-45b0-8b56-c9970e8fe1fa.png)


- The accuracy is around 93%.
- The precision for high risk loans is 9%. 
- The recall for high risk loans is 92%.

We see that the results obtained from this method is vastly improved for all the metrics.


## Summary

It has been mentioned above that accuracy alone is not a reliable metric when working with classification problems that are imbalanced. Accuracy is defined as the percentage of right predictions made by the algorithm (TP + TN)/(TP + TN + FP + FN).

where:
- TP is the True Positive
- TN is the True Negative
- FP is the False Positive
- FN is the False Negative

For data such as this where good loans vastly outnumber risky loans, the accuracy may be made 99.9% if all loans are assumed to be good loans. However any lending agency worth it's salt will want to identify possible defaulters, and therefore accuracy by itself cannot be considered a good measure of performance of classification models.\
Therefore other measures such as precision, and recall (or sensitivity), and the F1 score are considered, which are obtained from the classification report (we did not address F1 score for each of the above models but will touch upon here).\
Precision is the measure of the percentage predicted correctly as true positive by the model i.e. TP/(TP+FP).\
Recall is a measure of the percentage of actual true positives i.e. TP/(TP + FN).\
F1 score is the harmonic mean between precision and recall. A high F1 score implies a good balance between the two. In contrast a low score means that one of measures (precision or recall) is high at the expense of the other. We observe this for high risk loans in the classification report obatained from each of the ML algorithms.

In the present scenario, the Lending Club will benefit most if it is able to accurately identify most of the defaulters. This may come at a cost of missed opportunities with low risk clients. However incorrectly identified low risk clients may be further  scrutinized and re-evaluated correctly. In contrast incorrectly identifying defaulters as low risk may result in huge financial losses.

From the above analysis we see that the model that gives the best recall for high risk loans is the EasyEnsembleClassifier (an AdaBoost algorithm). The sensitivity is 92% meaning this model is able to catch 92% of the high risk loans in the dataset.\
The precision is 9% which means that out of all the high risk predictions made by the model, only 9% was correctly classified, and the remaining were misclassified. 9% is better than the precision measures from the other algorithms, but still low. However, the important measure to consider is the recall, as it's essential to capture most of the defaulters to avoid financial loss.

As far as the low risk predictions go, all the models gave a precision of 100%. A 100% for low risk loans and a low precision for high risk loans is expected from a dataset where low risk loans largely outnumbers the high risk loans.\
Resampling algorithms gave low recall for low risk loans around (40-69)% with the worst from the undersampling  method, ClusterCentroids (40%).\
Ensemble methods performed better with BalancedRandomForestClassifier giving a recall of 87%. The best was obtained from the EasyEnsembleClassifier, 94%

F1 score is a measure of the harmonic mean between precision and recall. Again the EasyEnsembleClassifier gave a high value of F1 score for the total, 97%. The BalancedRandomForestClassifier was also very good at 93%. All the re-sampling methods gave similar F1 score, around 75%, except for the undersampling ClusterCentroids algorithm at 56%.

From the above analysis, it would appear that overall the ensemble methods performed better at predicting credit risk compared to re-balancing the data outcomes using resampling methods. The worst performance was by the undersampling algorithm, ClusterCentroids.

**Is there a reccomendation on which model to use**?\
The ensemble machine learning model using AdaBoost, EasyEnsembleClassifier is reccomended for credit risk analysis because it accurately predicts the defaulters to a large percentage (92%). Its measures on low risk loans are also high, along with the total accuracy of the model. The total F1 score is 0.97 signalling a good balalnce between the precision and recall for low risk loans.
