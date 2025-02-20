import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf

# Load the model and preprocessor
model = tf.keras.models.load_model('deep_learning_model.keras')
preprocessor = joblib.load('preprocessor_pipeline.sav')

# Function to make predictions
def predict(item_weight, item_sugar_content, item_visibility, item_type, item_price, store_start_year, store_size, store_location_type, store_type):
    # Create a DataFrame with the input features
    data = pd.DataFrame({
        'Item_Weight': [item_weight],
        'Item_Sugar_Content': [item_sugar_content],
        'Item_Visibility': [item_visibility],
        'Item_Type': [item_type],
        'Item_Price': [item_price],
        'Store_Start_Year': [store_start_year],
        'Store_Size': [store_size],
        'Store_Location_Type': [store_location_type],
        'Store_Type': [store_type]
    })
    
    # Preprocess the data
    data_processed = preprocessor.transform(data)
    
    # Make prediction
    prediction = model.predict(data_processed)[0][0]
    return prediction

# Streamlit app layout
st.title('Item Store Returns Prediction')

# Input fields
item_weight = st.number_input('Item Weight', min_value=0.0, step=0.01)
item_sugar_content = st.selectbox('Item Sugar Content', ['Low', 'Medium', 'High'])
item_visibility = st.number_input('Item Visibility', min_value=0.0, step=0.01)
item_type = st.selectbox('Item Type', [
    'Baking Goods', 'Breads', 'Breakfast', 'Canned', 'Dairy', 'Frozen Foods', 'Fruits and Vegetables',
    'Hard Drinks', 'Health and Hygiene', 'Household', 'Meat', 'Others', 'Seafood', 'Snack Foods',
    'Soft Drinks', 'Starchy Foods'
])
item_price = st.number_input('Item Price', min_value=0.0, step=0.01)
store_start_year = st.number_input('Store Start Year', min_value=1900, max_value=2024, step=1)
store_size = st.selectbox('Store Size', ['small', 'medium', 'large'])
store_location_type = st.selectbox('Store Location Type', ['Cluster1', 'Cluster2', 'Cluster3'])
store_type = st.selectbox('Store Type', ['Supermarket1', 'Supermarket2', 'Grocery'])

# Prediction button
if st.button('Predict'):
    result = predict(item_weight, item_sugar_content, item_visibility, item_type, item_price, store_start_year, store_size, store_location_type, store_type)
    st.write(f'Predicted Item Store Returns: {result}')
