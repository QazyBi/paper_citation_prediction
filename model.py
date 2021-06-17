import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import Pool, CatBoostRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np


corpus = pd.read_csv('corpus.csv', index_col='id')
values = {'venue': corpus['venue'].mode()[0],
          'journalName': corpus['journalName'].mode()[0],
          'fieldsOfStudy': corpus['fieldsOfStudy'].mode()[0]
         }

corpus.fillna(value=values,inplace=True)
print(corpus.isna().sum())

y = corpus['citations_n']
X = corpus.drop(columns=['citations_n'])

# split data to 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

cat_features = ['venue','journalName','fieldsOfStudy']

train_pool = Pool(X_train, 
                  y_train, 
                  cat_features=cat_features)
test_pool = Pool(X_test, 
                 cat_features=cat_features) 

model = CatBoostRegressor(loss_function='RMSE', eval_metric='R2')
#train the model
model.fit(train_pool, verbose=100)
# make the prediction using the resulting model
preds = model.predict(test_pool)
print(preds)

print(f"r2_score: {r2_score(y_test, preds)}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.plot(y_test.value_counts() / len(y_test))
ax2.plot(y_train.value_counts() / len(y_train))
plt.show()


features = corpus.columns
importances = model.feature_importances_
indices = np.argsort(importances)

plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()