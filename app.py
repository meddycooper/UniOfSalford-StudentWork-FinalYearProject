# app.py

from flask import Flask, request, jsonify, render_template
from predict import predict_news

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # or a basic string for testing

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']
    prediction = predict_news(text)
    return jsonify({"prediction": prediction})

# Web browser form POST endpoint
@app.route('/predict_form', methods=['POST'])
def predict_form():
    text = request.form['text']
    prediction = predict_news(text)
    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)