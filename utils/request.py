"""

    Simple Script to test the API once deployed

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located at the root of this repo for guidance on how to use this
    script correctly.
    ----------------------------------------------------------------------

    Description: This file contains code used to formulate a POST request
    which can be used to develop/debug the Model API once it has been
    deployed.

"""

# Import dependencies
import requests
import pandas as pd
import numpy as np
from sklearn import preprocessing

# Load data from file to send as an API POST request.
# We prepare a DataFrame with the public test set 
# from the Kaggle challenge.
feature_vector_df = pd.read_csv('data/test_data.csv')
feature_vector_df = feature_vector_df[(feature_vector_df['Commodities'] == 'APPLE GOLDEN DELICIOUS')]
predict_vector = feature_vector_df
predict_vector = predict_vector.reset_index(drop=True)
predict_vector.drop('Commodities',axis = 1, inplace = True)   
predict_vector['Date'] = predict_vector['Date'].apply(lambda x: pd.to_datetime(x))
predict_vector['Day'] = predict_vector['Date'].dt.day
predict_vector['Month'] = predict_vector['Date'].dt.month
predict_vector['Year'] = predict_vector['Date'].dt.year
predict_vector.drop('Date', axis = 1, inplace = True)   
label_encoder = preprocessing.LabelEncoder()

# Encode labels in column 'Size_Grade' in training data. 
predict_vector['Size_Grade']= label_encoder.fit_transform(predict_vector['Size_Grade']) 

# Encode labels in column 'Container' in training data
predict_vector['Container']= label_encoder.fit_transform(predict_vector['Container'])   
test = pd.get_dummies(predict_vector, drop_first=True)   

# Convert our DataFrame to a JSON string.
# This step is necessary in order to transmit our data via HTTP/S
feature_vector_json = test.iloc[1].to_json()

# Specify the URL at which the API will be hosted.
# NOTE: When testing your instance of the API on a remote machine
# replace the URL below with its public IP:

# url = 'http://{public-ip-address-of-remote-machine}:5000/api_v0.1'
url = 'http://127.0.0.1:5000/api_v0.1'


# Perform the POST request.
print(f"Sending POST request to web server API at: {url}")
print("")
print(f"Querying API with the following data: \n {test.iloc[1].to_list()}")
print("")
# Here `api_response` represents the response we get from our API
api_response = requests.post(url, json=feature_vector_json)

# Display the prediction result
print("Received POST response:")
print("*"*50)
# print(f"API prediction result: {api_response.json()[0]}")
print(f"API prediction result: {api_response.json()}")
print(f"The response took: {api_response.elapsed.total_seconds()} seconds")
print("*"*50)
