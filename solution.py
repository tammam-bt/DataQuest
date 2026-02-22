import pandas as pd
import numpy as np
import joblib

def preprocess(df_input):
    df = df_input.copy()
    if 'Employer_ID' in df.columns:
        df.drop('Employer_ID', axis=1, inplace=True)

    # Broker & Region Logic
    df['Broker_ID'] = df['Broker_ID'].fillna('Unknown')
    broker_counts = df['Broker_ID'].value_counts()
    df['Broker_Policy_Count'] = df['Broker_ID'].map(broker_counts)
    df['Region_Code'] = df['Region_Code'].fillna('Unknown')
    df['Region_Code_Enc'] = df['Region_Code'].astype('category').cat.codes
    
    # Categories
    df['Broker_Agency_Type_Bin'] = (df['Broker_Agency_Type'] == 'National_Corporate').astype(int)
    df['Deductible_Tier'] = df['Deductible_Tier'].fillna("Tier_1_High_Ded")
    deductible_mapping = {'Tier_1_High_Ded': 1, 'Tier_2_Mid_Ded': 2, 'Tier_3_Low_Ded': 3, 'Tier_4_Zero_Ded': 4}
    df['Deductible_Tier_Num'] = df['Deductible_Tier'].map(deductible_mapping)

    # Dummies (This is usually where column mismatches happen)
    df['Acquisition_Channel'] = df['Acquisition_Channel'].fillna('Unknown')
    df = pd.get_dummies(df, columns=['Acquisition_Channel'], prefix='Acq')

    # Encodings
    df['Payment_Schedule_Enc'] = df['Payment_Schedule'].astype('category').cat.codes
    df['Employment_Status_Enc'] = df['Employment_Status'].astype('category').cat.codes

    # Month
    month_mapping = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 
                     'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}
    df['Policy_Start_Month_Num'] = df['Policy_Start_Month'].map(month_mapping).fillna(0)

    return df

def load_model():
    # Load the joblib file
    return joblib.load('model.joblib')

def predict(df_processed, model):
    user_ids = df_processed['User_ID'].copy()
    
    # Get the features the model expects
    try:
        expected_features = model.feature_names_
    except AttributeError:
        # Fallback if names aren't embedded
        expected_features = [col for col in df_processed.columns if col not in ['User_ID', 'Purchased_Coverage_Bundle']]

    X = df_processed.copy()
    
    # Ensure all expected columns exist
    for col in expected_features:
        if col not in X.columns:
            X[col] = 0
            
    # Reorder to match model exactly
    X = X[expected_features]

    # Generate predictions
    y_pred = model.predict(X)
    y_pred = np.array(y_pred).flatten().astype(int)

    return pd.DataFrame({
        'User_ID': user_ids,
        'Purchased_Coverage_Bundle': y_pred
    })