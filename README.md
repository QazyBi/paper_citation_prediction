# Predict Number of Citations of a Paper

Main question: How can we predict number of citations of a paper before it has been published?

## 1. Parsing data
- data has been downloaded from the storage as .gz format 
- there are 30 archived files with about 60 MB unarchived

## 2. Data pre-processing
- dataset is about **950k** rows
- most of the columns of type object
- h-index, g-index columns were created from citation numbers 

## 3. Data Analysis
observations:
- data is imbalanced, most of the papers have 0 or 1 citations

## 4. Predict the number of citations

model selection:
- xgboost as it allows to gain more performance with no serious time penalty


gridSearch: diagram
r2-score: 0.8865324955807499

best parameters:
{
 'colsample_bytree': 0.7,
 'learning_rate': 0.07,
 'max_depth': 6,
 'min_child_weight': 4,
 'n_estimators': 500,
 'nthread': 4,
 'objective': 'reg:linear',
 'silent': 1,
 'subsample': 0.7
 }

feature importance: diagram



## 5. Conclusion
In this project, I analyzed metadata of research papers. The data consist various information like: authors, year of publishing, paper abstract, citations, journal name, field of study, link and etc. There were about a million paper data in the dataset, I had to use all the necessary features to predict how many citations new papers will have.

I retrieved the data from the source. Used pandas dataframes to work with data. After preprocessing steps, I analyzed factors that correlated with number of citations, and did some exploratory visualization and analysis.

When the data was cleaned and features selected I created boosting model which uses predictions of several simple models to predict number of citations paper will have.  

You can refer to the `task.ipynb` file to find more. 

Side-note
If I am using non-public data, and you are the author, please contact me.
