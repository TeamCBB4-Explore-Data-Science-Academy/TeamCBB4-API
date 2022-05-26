"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
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
    df1 = feature_vector_df
    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------
    #predict_vector = feature_vector_df[['Madrid_wind_speed','Bilbao_rain_1h','Valencia_wind_speed']]
    ''''
    df1['time']=pd.to_datetime(df1['time'], infer_datetime_format=True) 
    df1['time']= pd.to_numeric(df1['time'].dt.strftime("%Y%m%d%H%M%S"))
    
    q = df1['Valencia_pressure'].mean()
    df1['Valencia_pressure'].fillna(value=q, inplace=True)
    dataMapping = {'sp1':1, 'sp2':2, 'sp3':3, 'sp4':4, 'sp5':5, 'sp6':6, 'sp7':7, 'sp8':8, 'sp9':9, 'sp10':10,
             'sp11':11, 'sp12':12, 'sp13':13, 'sp14':14, 'sp15':15, 'sp16':16, 'sp17':17, 'sp18':18, 'sp19':19, 
             'sp20':20, 'sp21':21, 'sp22':22, 'sp23':23, 'sp24':24, 'sp25':25} 
    df1['Seville_pressure_num'] = df1['Seville_pressure'].map(dataMapping)
    del df1['Valencia_wind_deg']
    del df1['Seville_pressure']
    '''
    #df1['time'] = pd.to_datetime(df1.time)
    #df1['time'] = (df1['time'] - df1['time'].min())  / np.timedelta64(1,'D')



    df1 = pd.DataFrame.from_dict([feature_vector_dict])
    for i in df1.columns.to_list():
        if (df1[i].dtype =='float64' or df1[i].dtype =='int64'):
           df1[i] = df1[i].fillna(value=df1[i].mean())
        else:
            df1[i] = df1[i].fillna(value=df1[i].mode())
    df1['time']=pd.to_datetime(df1['time'], infer_datetime_format=True) 
    df1['time']= pd.to_numeric(df1['time'].dt.strftime("%Y%m%d%H%M%S"))
    q = df1['Valencia_pressure'].mean()
    df1['Valencia_pressure'].fillna(value=q, inplace=True)
    dataMapping = {'sp1':1, 'sp2':2, 'sp3':3, 'sp4':4, 'sp5':5, 'sp6':6, 'sp7':7, 'sp8':8, 'sp9':9, 'sp10':10,
             'sp11':11, 'sp12':12, 'sp13':13, 'sp14':14, 'sp15':15, 'sp16':16, 'sp17':17, 'sp18':18, 'sp19':19, 
             'sp20':20, 'sp21':21, 'sp22':22, 'sp23':23, 'sp24':24, 'sp25':25} 
    df1['Seville_pressure_num'] = df1['Seville_pressure'].map(dataMapping) 
    del df1['Valencia_wind_deg']
    del df1['Seville_pressure'] 

    for i in df1.columns.to_list():
        if int(df1[i].isnull().sum()) != 0:
            df1.drop([i], axis=1, inplace = True)


    #from sklearn.model_selection import train_test_split
    #y=df.load_shortfall_3h
    #X=df.drop(['load_shortfall_3h'],axis=1)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3332,train_size = 0.6668, random_state=50)
    # ------------------------------------------------------------------------
    return df1
    #return predict_vector

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


""" You may use this section (above the make_prediction function) of the python script to implement 
    any auxiliary functions required to process your model's artifacts.
"""

def make_prediction(data, model):
    """Prepare request data for model prediction.

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
    # Format as list for output standardisation.
    return prediction[0].tolist()
