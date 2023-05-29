# Report: Predict Bike Sharing Demand with AutoGluon Solution
#### Bilal Ahmed

## Initial Training
### What did you realize when you tried to submit your predictions? What changes were needed to the output of the predictor to submit your results?
 Predictions needed to be greater than or equal to zero. That's why I have to replace negative predictions to zero.

### What was the top ranked model that performed?
We can use predictor.leaderboard() to see the all models, in my case WeightedEnsemble_L3 is the best.

## Exploratory data analysis and feature creation
### What did the exploratory analysis find and how did you add additional features?
datetime fieled was object which was converted into datetime and then hour, day, month an year was extracted from that single column
moreover, season and weather were integer features but converted to categories

### How much better did your model preform after adding additional features and why do you think that is?
Change in performace was amazing inital score was 1.778 and after adding new features it went down to 0.728.
Reason of this big change was because of better representation of datetime column it was containing much information such as hour, day, month and year. This helps model find pattern in the data as this data is time sensitive. However, converting season and weather to categorical features also help model to find better patterns.

## Hyper parameter tuning
### How much better did your model preform after trying different hyper parameters?
From 0.728 score it went to 0.553 which is not the huge difference as I have not tuned it longer and I barely tuned three models only.

### If you were given more time with this dataset, where do you think you would spend more time?
I think I would spend more time on creating new features that will definitely help increase score.

### Create a table with the models you ran, the hyperparameters modified, and the kaggle score.
|model|hpo1|score|
|--|--|--|
|initial|default|1.778|
|add_features|default|0.728|
|hpo|{'GBM': [{'extra_trees': True, 'min_data_in_leaf': 25, 'num_boost_round': 100}, {'extra_trees': True, 'min_data_in_leaf': 30, 'num_boost_round': 150}, {}, 'GBMLarge'], 'RF': [{'criterion': ['gini', 'entropy', 'log_loss'], 'n_estimators': 115}, {'criterion': ['gini', 'entropy', 'squared_error'], 'n_estimators': 150}], 'KNN': [{'weights': 'uniform', 'leaf_size': 35, 'ag_args': {'name_suffix': 'Unif'}}, {'weights': 'distance', 'leaf_size': 40, 'ag_args': {'name_suffix': 'Dist'}}]}|0.553|

### Create a line plot showing the top model score for the three (or more) training runs during the project.

![Train Scores New Features](cd0385-project-starter/project/new_features_score.png)

### Create a line plot showing the top kaggle score for the three (or more) prediction submissions during the project.

![Model Scores Kaggle](cd0385-project-starter/project/model_test_score.png)

## Summary
In this project, we have explored following steps in the ML Lifecycle.

* Product: Business use case understanding is the most crucial part, if problem is not nailed chances of moving is wrong direction is eminent.

* Data: It includes understanding of data gathering and processing, in this case we used built-in open source data from kaggle competition.

* Engineering: It includes exploratary data analysis which works as fuel for modeling part. Then modeling and hyperparamter tuning is also part of this stage.

* Results: Saved models can be used to get inference or ploting of result for story telling.

