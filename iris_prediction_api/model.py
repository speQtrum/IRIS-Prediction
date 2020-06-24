import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle



# Importing the dataset
dataset = pd.read_csv('webapp_103/iris.csv')
X = dataset.iloc[:,0:4].values
y = dataset.iloc[:, 4].values

classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X,y)
pickle.dump(classifier, open('model.pkl','wb'))





# Predicting the Test set results
print(classifier.predict([[4.6,3.6,1,0.2]]))
# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[6.2,2.2,4.5,0.2]]))