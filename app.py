import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(layout="wide")
st.title("Fish Weight Prediction") 

# Reading the data from a CSV file
Dataset = pd.read_csv('Fish.csv')

# Creating a dictionary to map fish species to numerical values
category = {
    "Bream": 1,
    "Roach": 5,
    "Whitefish": 7,
    "Parkki": 2,
    "Perch": 3,
    "Pike": 4,
    "Smelt": 6,
}

# Dropdown box for selecting fish species
options = ['Bream', 'Roach', 'Whitefish', 'Parkki', 'Perch', 'Pike', 'Smelt']
Species = st.selectbox('Select Fish Species:', options)

# Input fields for other features
Height = st.number_input('Enter the height in cm:', step=0.1)
Width = st.number_input('Enter the width in cm:', step=0.1)
Vertical_length = st.number_input('Enter the length1 in cm:', step=0.1)
Diagonal_length = st.number_input('Enter the Length2 in cm:', step=0.1)
Cross_length = st.number_input('Enter the Length3 in cm:', step=0.1)

# Predict fish weight when the Predict button is clicked
if st.button("Predict"):
    # Check if any required field is empty
    if (Height == 0 or Width == 0 or Vertical_length == 0 or
            Diagonal_length == 0 or Cross_length == 0):
        st.error("Please fill in all required fields.")
    else:
        # Selecting features and target variable
        X = Dataset[['Category', 'Height', 'Width', 'Length1', 'Length2', 'Length3']]
        y = Dataset['Weight']

        # Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=2529)

        # Creating the Random Forest Regressor model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Convert inputs to numeric values
        user_input = [category[Species], Height, Width, Vertical_length, Diagonal_length, Cross_length]

        # Predicting fish weight
        Predicted_data = model.predict([user_input])
        st.success(f"The estimated 🐟 weight is {int(Predicted_data[0])} grams.")

# Footer content
footer_html = """
<hr>
<div style="bottom: 0;  color: white; text-align: center;">
    <p style="font-weight: bold; ">Developed by Santhoshi Kumari Kothapalli</p>
</div>
"""

# Display the footer using markdown
st.markdown(footer_html, unsafe_allow_html=True)        
