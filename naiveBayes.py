import pandas as pd
import numpy as np
df = pd.read_csv("Social_Network_Ads.csv")

gender_df=pd.get_dummies(df['Gender'],drop_first=True)
df.drop('User ID',axis=1,inplace=True)
df.drop('Gender',axis=1,inplace=True)

df=pd.concat([df,gender_df],axis=1)

X = df.iloc[:, [0, 1,3]].values
y = df.iloc[:, -2].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

import pickle
pickle.dump(sc, open("scaler.pickle", "wb"))
ssc = pickle.load(open("scaler.pickle", 'rb')) 

pickle.dump(classifier, open('nbclassifier.pkl','wb'))

model = pickle.load(open('nbclassifier.pkl','rb'))