from flask import Flask, jsonify, request, render_template
from typing import Dict
import numpy as np
import pickle


model = pickle.load(open("rfc.pkl", 'rb'))
model2 = pickle.load(open("retrained_model.pkl", 'rb'))

app = Flask(__name__)

numeric_features = ['artist_familiarity', 'artist_hotttnesss', 'duration', 'end_of_fade_in', 'key',
                    'key_confidence', 'loudness', 'mode', 'mode_confidence', 'start_of_fade_out', 'tempo',
                    'time_signature', 'time_signature_confidence', 'year', 'artist_latitude', 'artist_longitude']
label = ['popularity']

@app.route('/')
def hello_world():
    return render_template('home.html')


@app.route('/predict', methods=['POST','GET'])
def predict():
    input_values = []
    for feature in numeric_features:
        input_value = float(request.form[feature])
        print(input_value)
        input_values.append(input_value)
    # Convert the input values to a numpy array
    input_array = np.array(input_values).reshape(1, -1)
    print(input_array)
    # Make a prediction with the model
    prediction1 = model.predict(input_array)
    prediction2 = model2.predict(input_array)

    print(prediction1)
    popularity1 = round(prediction1[0], 2)
    popularity2 = round(prediction2[0], 2)
    return render_template('home.html', prediction_text='Predicted popularity: {}'.format(popularity1),prediction_text_retrained='Predicted popularity retrained: {}'.format(popularity2))



if __name__ == "main":
    app.run(debug=True)


