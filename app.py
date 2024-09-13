import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
!pip install -q streamlit
import streamlit as st
import pickle

# Load the dataset
file_path = "/content/sample_data/CED.xlsx"
data = pd.read_excel(file_path)

# Inspect the first few rows
print(data.head())

X = data.drop('class', axis=1)
y = data['class']
X.head() 

# Convert categorical features to numerical using one-hot encoding
X = pd.get_dummies(X, drop_first=True)

# train data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

#Implement Random Forest classifier
classifier = RandomForestClassifier()

classifier.fit(X_train,y_train)

## Prediction
y_pred = classifier.predict(X_test)

### Check Accuracy

score = accuracy_score(y_test,y_pred)

score

# Streamlit interface
st.title("Car Class Prediction")

# Add a greeting message with emoji
st.write('Welcome to the *Car Class Prediction App!* :car: :chart_with_upwards_trend:')

# Create input fields for user
buying = st.selectbox('Buying', ['low', 'med', 'high', 'vhigh'])
maint = st.selectbox('Maint', ['low', 'med', 'high', 'vhigh'])
doors = st.selectbox('Doors', [2, 3, 4, 5])
persons = st.selectbox('Persons', [2, 4, 'more'])
lug_boot = st.selectbox('Lug Boot', ['small', 'med', 'big'])
safety = st.selectbox('Safety', ['low', 'med', 'high'])

# Create a dictionary of inputs
input_data = {
    'buying': buying,
    'maint': maint,
    'doors': doors,
    'persons': persons,
    'lug_boot': lug_boot,
    'safety': safety
}

# When the user clicks 'Predict'
if st.button('Predict'):
    result = predict_class(input_data)
    st.write(f"The predicted car class is: {result}")


pickle_out = open("prediction.pkl","wb")

pickle.dump(classifier, pickle_out) 
pickle_out.close()
