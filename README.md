
# Titanic-Machine-Learning-from-Disaster 

Overview

The data has been split into two groups:

* training set (train.csv)
* test set (test.csv)

The training set should be used to build your machine learning 
models. For the training set, we provide the outcome (also known 
as the “ground truth”) for each passenger. Your model will be 
based on “features” like passengers’ gender and class. You can 
also use feature engineering to create new features.

The test set should be used to see how well your model performs 
on unseen data. For the test set, we do not provide the ground 
truth for each passenger. It is your job to predict these 
outcomes. For each passenger in the test set, use the model you 
trained to predict whether or not they survived the sinking of 
the Titanic.

We also include gender_submission.csv, a set of predictions that 
assume all and only female passengers survive,
## Data Analysis

In Titanic-Machine-Learning-from-Disaster dataset we have 11 features(including target 
variable) and 891 records in training data and in test data we have 
10 variable(excluding target variable) and 418 records .

* PassengerID : ID of the Passenger.
* Survived : No (0) Yes (1).
* Pclass : Passenger’s class.
* Name : Name of the Passenger.
* Sex : Gender of the Passenger.
* SibSp : Number of Siblings / Spouses aboard.
* Parch : Number of Parents / Children aboard.
* Ticket : Ticket number.
* Fare : Fare of the Ticket.
* Cabin : Cabin number.
* Embarked : Port of embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
## Approach

Our Main goal is to know that whether which check whether the person is survived or not-survived.

* Data Exploration : Exploring dataset using pandas, numpy, matplotlib and seaborn.
* Data visualization :Ploted graphs to get insights about dependend and independed variables.
* Model Selection I : Tested all base models to check the base accuracy. Also ploted and calculate Performance Metrics to check whether a model is a good fit or not.
* Model Selection II : After testing all base if some of them are not working properly then we have to do model tunning. 
* Pickle File : Selected model as per best accuracy and created pickle file using pickle library.
* Webpage & deployment : Created a webform that takes all the necessary inputs from user and shows output. After that I have deployed project on Herokuapp. 

## Technologies Used

* Jupyter notebook, Spyder, VScode Is Used For IDE.
* For Visualization Of The Plots Matplotlib , Seaborn Are Used.
* Herokuapp is Used For Model Deployment.
* Front End Deployment Is Done Using HTML, CSS, Bootstrap.
* Flask is for creating the application server and pages.
* Git Hub Is Used As A Version Control System.
* os is used for creating and deleting folders.
* csv is used for creating .csv format file.
* numpy is for arrays computations and mathematical operations
* pandas is for Manipulation and wrangling structured data
* scikit-learn is used for machine learning tool kit
* pickle is used for saving model
* Logistic regression is used for LogisticRegression Implementation.
* SGD is used for SGDClassifier Implementation.
* K-Nearest Neighbors is used for KNeighborsClassifier Implementation.
* SVM is used for SVC Implementation.
* Decision Tree is used for DecisionTreeClassifier Implementation.
* Extra Trees is used for ExtraTreesClassifier Implementation.
* Random forest is used for RandomForestClassifier Implementation.
* Ada boost is used for AdaBoostClassifier Implementation.
* Gradient is used for GradientBoostingClassifier Implementation.
* XGboost is used for XGBClassifier Implementation
* Ensemble is used for VotingClassifier Implementation.
## Deployment

**Herokuapp Link:** https://titanic-ml-from-disaster.herokuapp.com/
  
## Deployment

To Clone this project run

```bash
git clone https://github.com/vish-68/Titanic-Machine-Learning-from-Disaster
```
Go to Project directory
```bash
cd Titanic-Machine-Learning-from-Disaster
```
Install dependencies
``` bash
pip install -r requirements.txt
```  
Run the app.py
```bash
python app.py
```
## Conclusion

We developed Pima Indians Diabeties model which is capable to predict
whether the patient is diabetic or non-diabetic. In this dataset 9 features
(including target variable) and 768 records.
* Our 1st step is to import dataset and check all
  the details like shape, info(), describe() for getting better knowledge
  about data or each variable.

* 2nd step is to checking missing value and datatype of missing variable
  by looking at info(). after that we have to delete those 
  variable whose missing value is more than 50% of data. Other 
  variable should be treat as repect to their datatype

* 3rd step After handling all this we have to do data 
  visualization for getting some insight for eg. we have to check 
  for ouliers, imbalanced variable or imbalanced data so we have to 
  remove ouliers or do upsampling for those. In this there is no 
  need to treat oulier so after doing visualization we can say 
  that maximum people who died is from class 3 and count of 
  male died is more as compared to female.

* 4th step is to do Preprocessing using labelencoder and perform 
  StandardScalar for scaling data

* 5th step Splitting data into training and validation data and building 
  different ML model like LogisticRegression, SGDClassifier, 
  KNeighborsClassifier, SVR, DecisionTreeRClassifier, ExtraTreesClassifier, 
  RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, 
  VotingClassifier, XGBClassifier. Out of all KNN Tune model is working 
  properly as compared to others.

* 6th step is do perform all the operation which is perform in training
  data till Preprocessing part after that while scaling data we have 
  only transform data, we do not have to perform fit()

* 7th step is we have to us predict() and save model in .pkl file.

* Last step is model Deployment using Flask Framework.
  For model deployment we have to dump our model using pickle library
  and can save our model in .pkl or .sav extension.
