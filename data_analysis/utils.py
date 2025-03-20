from config import *

# Imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os

from scipy.linalg import LinAlgError
import pickle, itertools

# Methods
def import_from_files_single_dataframe(rootdir): # read data from file
    file_list =[]
    for dirpath, dirnames, filenames in os.walk(rootdir):
        for file in filenames:
            filename = os.path.join(dirpath, file)
            print(filename)
            file_list.append(filename)

    df = pd.concat((pd.read_csv(f) for f in file_list), ignore_index=True)

    return df

# Function to serialise an object into a pickle file
def save_to_pickle(file_name, export_folder, save_data, complete_path=True):
    file_name_with_extension = file_name + ".pkl"
    complete_file_path = f'{export_folder}/{file_name_with_extension}' if(complete_path) else file_name
    with open(complete_file_path, 'wb') as file:
        pickle.dump(save_data, file)

def load_from_pickle(file_name, export_folder, complete_path=True):
    if(".pkl" not in file_name):
        file_name_with_extension = file_name + ".pkl"
    else:
        file_name_with_extension = file_name
    complete_file_path = f'{export_folder}/{file_name_with_extension}' if(complete_path) else file_name
    with open(complete_file_path, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data

def plot_histogram(df): # Function to plot feature-NLOS histogram (Code entirely lifted from [4] https://github.com/ptrpfa/UWB-LOS-NLOS-Classification/tree/main)
    # Get non-class features
    features = [col for col in df.columns if col != 'NLOS']
    plt.figure(figsize=(20, 20))
    for i, feature in enumerate(features, start=1):
        plt.subplot(len(features)//2 + 1, 2, i)
        try:
            sns.histplot(data=df, x=feature, hue='NLOS', kde=True, stat='density', common_norm=False)
        except LinAlgError as e:
            # print(f"Warning: {e}")
            sns.histplot(data=df, x=feature, hue='NLOS', stat='density', common_norm=False)
        plt.title(f'Distribution of {feature} by NLOS')
        plt.xlabel(feature)
        plt.ylabel('Density')
        los_skewness = df[df['NLOS'] == 0][feature].skew()
        nlos_skewness = df[df['NLOS'] == 1][feature].skew()
        plt.text(0.9, 0.9, f'Skewness (LOS): {los_skewness:.2f}\nSkewness (NLOS): {nlos_skewness:.2f}', 
                 horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes)
    plt.tight_layout()
    plt.show()

def plot_scatter(df): # Function to plot feature-RANGE scatterplot (Code entirely lifted from [4] https://github.com/ptrpfa/UWB-LOS-NLOS-Classification/tree/main)
    # Get non-class features
    features = [col for col in df.columns if col != 'RANGE']
    plt.figure(figsize=(20, 20))
    for i, feature in enumerate(features, start=1):
        plt.subplot(len(features)//2 + 1, 2, i)
        try:
            sns.scatterplot(data=df, x=feature, y='RANGE')
        except LinAlgError as e:
            # print(f"Warning: {e}")
            sns.scatterplot(data=df, x=feature, y='RANGE')
        plt.title(f'Distribution of {feature} by RANGE')
        plt.xlabel(feature)
        plt.ylabel('RANGE')
        # los_skewness = df[df['RANGE'] == 0][feature].skew()
        # nlos_skewness = df[df['RANGE'] == 1][feature].skew()
        # plt.text(0.9, 0.9, f'Skewness (LOS): {los_skewness:.2f}\nSkewness (RANGE): {nlos_skewness:.2f}', 
        #          horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes)
    plt.tight_layout()
    plt.show()

def get_top_features(df):
    X = df.drop(columns='NLOS')
    y = df['NLOS']

    # Initialize the Random Forest Classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)

    # Fit the model
    rf_classifier.fit(X, y)

    # Get feature importances
    feature_importances = rf_classifier.feature_importances_

    # Create a DataFrame for visualization
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})

    # Sort the DataFrame based on feature importance
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Display the top 20 most important features
    # return feature_importance_df
    return feature_importance_df

    