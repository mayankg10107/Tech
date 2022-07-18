"""
Testing File
"""
import pandas as pd
from sklearn.metrics import mean_squared_error
import joblib


def testing(filename):
    """
    Function to test the model
    filename: path to the training data file
    Function returns the mean_squared_error of the predicted results
    """

    # Loading the saved model
    try:
        model = joblib.load("model.pkl")
    except FileNotFoundError:
        return "Model not found or Error loading model, try retraining and resaving the model"

    # Reading the csv file
    try:
        test_data = pd.read_csv(filename, index_col="ID")
    except FileNotFoundError:
        return "File not found or Error reading .csv file"

    # After loading the data, explicitly define the feature matrix and the target variable vector.
    try:
        x_test = test_data.loc[:, ["height", "weight", "age"]]
        y_test = test_data.loc[:, ["BicepC"]]
    except KeyError:
        return "Please check the contents of the file"

    # Prediction
    try:
        y_pred = model.predict(x_test)
    except ValueError:
        return "Error predicting"

    return mean_squared_error(y_test, y_pred)
