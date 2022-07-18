"""
import library you made (regression) and 3rd party library here
"""
import os
from flask import Flask, request, render_template
from regression.train import training
from regression.test import testing


app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = ""
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/")
def index():
    """
    Home
    """
    return render_template("home.html")


@app.route("/", methods=["POST"])
def train():
    """
    Function to load the data and
    print the model coefficients, intercept and mean squared error
    """
    uploaded_file = request.files["file"]
    print("Requested file is ")
    print(uploaded_file)
    if uploaded_file.filename != "":
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], uploaded_file.filename)
        uploaded_file.save(file_path)
        coefficients, intercept, mean_sq_error = training(file_path)
        print(coefficients, intercept, mean_sq_error)
        results = (
            "Model trained succesfully. Coefficients of the model: "
            + str(coefficients[0])
            + ", Intercept: "
            + str(intercept[0])
            + ", Mean Squared Error: "
            + str(mean_sq_error)
        )
        os.remove(file_path)
    return render_template("train.html", data_var=results)


@app.route("/test_data", methods=["POST"])
def test_func():
    """
    Function to load the data and
    print the mean squared error
    """
    uploaded_file = request.files["file"]
    if uploaded_file.filename != "":
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], uploaded_file.filename)
        uploaded_file.save(file_path)
        mean_sq_error = testing(file_path)
        results = "The Mean Squared Error on the test data is: " + str(mean_sq_error)
        os.remove(file_path)

    return render_template("train.html", data_var2=results)


app.run()


# if __name__ == "__main__":
#     # implement your application code here
#     pass
