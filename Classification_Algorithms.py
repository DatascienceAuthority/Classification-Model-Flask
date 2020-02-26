import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

Path = "Carseats.csv"
data = pd.read_csv(Path)

data.loc[data.Sales > 4, 'Sale'] = 'Yes'
data.loc[data.Sales < 4, 'Sale'] = 'No'
data = data.loc[:,data.columns != 'Sales']

data['Sale'], sales_index = pd.factorize(data['Sale'])
sales_index
print(data['Sale'].unique())

data['ShelveLoc'], shelveloc_index = pd.factorize(data['ShelveLoc'])
shelveloc_index
data['Urban'], urban_index= pd.factorize(data['Urban'])
urban_index
data['US'], us_index = pd.factorize(data['US'])
us_index

data.info()


X = data.loc[:,data.columns != 'Sale']
Y = data.Sale

feature_names = X.columns

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

################# Random Forest ##################################
model = RandomForestClassifier(n_estimators=100)

model.fit(X_train,y_train)

predicted = model.predict(X_test)


print(metrics.confusion_matrix(y_test, predicted))

print(metrics.classification_report(y_test, predicted))

RF_accuracy = metrics.accuracy_score(y_test, predicted)
print('RF_Accuracy: {:.2f}'.format(RF_accuracy))

# save the model to disk
import pickle
filename = 'model/model.pkl'
pickle.dump(model, open(filename, 'wb'))

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print(result)

#testing
prediction = model.predict([[125, 87, 9, 232, 120, 1, 42, 10, 0, 0]])
prediction
prediction[0]
X_test