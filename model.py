import pandas as pd
import numpy as numpy
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# IMPORTING DATASETS
df = pd.read_csv('Social_Network_Ads.csv')

# X-Y SPLIT
x=df.drop(['User ID','Purchased'], axis=1)
y=df['Purchased']

# ONE HOT ENCODING
x=pd.get_dummies(x,columns=['Gender'],dtype= int)

# PRESERVING COLUMNS FOR LATER
feature_columns = x.columns.tolist()

# TRAIN- TEST SPLIT
x_train,x_test,y_train,y_test =train_test_split(
    x,y,test_size=0.2,random_state=42
)

# STANDARD SCALING X_TRAIN ANS X_TEST
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

# ML MODELLING
model = LogisticRegression()
model.fit(x_train,y_train)

# PREDICTION AND TESTING, ACCURACY
y_pred = model.predict(x_test)
print('Accuracy = ',accuracy_score(y_test,y_pred))

# SAVE THE MODEL - PICKLE
pickle.dump(model,open('model.pkl','wb'))            # convert the model to bits and store it as pickle
# SAVE THE SCALER
pickle.dump(scaler,open('scaler.pkl','wb'))
# SAVE THE FEATURES
pickle.dump(feature_columns,open('feature_columns.pkl','wb'))

print('Model Training Completed')
