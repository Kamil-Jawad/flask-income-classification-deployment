# from flask import Flask
from flask import Flask, render_template, request
import pandas as pd
import pickle


def income_classs_prediction(k1, k2, k3, k4, k5, k6, k7):

    #   deserialized or open the scaler
    scaler = pickle.load(open('scaler.pkl', 'rb'))

    #   deserialized or open trained model
    model = pickle.load(open('model.pkl', 'rb'))

#   convert variables into integers
    k1 = int(k1)
    k2 = int(k2)
    k3 = int(k3)
    k4 = int(k4)
    k5 = int(k5)
    k6 = int(k6)
    k7 = int(k7)

#    prepare data for prediction
    prepare_data = {
        'age': [k1],
        'fnlwgt': [k2],
        'education-num': [k3],
        'sex': [k4],
        'capital-gain': [k5],
        'capital-loss': [k6],
        'hours-per-week': [k7]
    }

    prepare_data = pd.DataFrame(prepare_data)

#     normalized the prepare_data
    scaled_data = scaler.transform(prepare_data)

    # get prediction and save in variable
    predict_result = model.predict(scaled_data)


#     convert into senceable string
    if (predict_result[0] == 1):
        predict_result1 = "Salary is greater than 50K $"
        return predict_result1
    elif (predict_result[0] == 0):
        predict_result1 = "Salary is less or equal to 50K $"
        return predict_result1


app = Flask(__name__)


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/", methods=['POST'])
def getvalue():
    k1 = request.form['a1']
    k2 = request.form['a2']
    k3 = request.form['a3']
    k4 = request.form['a4']
    k5 = request.form['a5']
    k6 = request.form['a6']
    k7 = request.form['a7']

    prediction1 = income_classs_prediction(k1, k2, k3, k4, k5, k6, k7)
    # here data can be pass in other html files like "pass.html"
    return render_template("index.html", x1=prediction1)


if __name__ == "__main__":
    app.run(debug=False)
