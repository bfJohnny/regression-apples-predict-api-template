"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json
from sklearn import preprocessing

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.

    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------
    

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
    predict_vector = pd.get_dummies(predict_vector, drop_first=True)                   
    # ------------------------------------------------------------------------

    return predict_vector

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))

def make_prediction(data, model):
    """Prepare request data for model prediciton.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standerdisation.
    return prediction[0].tolist()
