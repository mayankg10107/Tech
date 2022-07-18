"""
Training File
"""
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib


def training(filename):
    """
    Function to train the model
    filename: path to the training data file
    Function returns the model coefficients and model intercept_, mean_squared_error
    """

    # filename = "data/regression_train.csv",
    try:
        train_data = pd.read_csv(filename, index_col="ID")
    except FileNotFoundError:
        return "File not found or Error reading .csv file"

    # After loading the data, explicitly define the feature
    # matrix and the target variable vector.
    try:
        x_train = train_data.loc[:, ["height", "weight", "age"]]
        y_train = train_data.loc[:, ["BicepC"]]
    except KeyError:
        return "Please check the contents of the file"

    # Define the linear regression model
    model = LinearRegression(
        fit_intercept=True, normalize=False, copy_X=True, n_jobs=None
    )

    # Fitting the model on the training data
    try:
        model.fit(x_train, y_train)
    except ValueError:
        return "Error fitting the model"

    # Predicting the trained data
    try:
        y_pred = model.predict(x_train)
    except ValueError:
        return "Error predicting"

    # Returing the trained model coefficients, intercept and the mean squared
    # error between the predicted and actual results
    try:
        joblib.dump(model, "model.pkl")
    except RuntimeError:
        return "Error saving the model"

    return model.coef_, model.intercept_, mean_squared_error(y_train, y_pred)
