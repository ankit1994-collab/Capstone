from flask import Flask, jsonify, request, render_template, jsonify
from typing import Dict
import numpy as np
import pandas as pd
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
    return render_template('upload.html')

@app.route('/process-csv', methods=['POST'])
def process_csv():
    # get the uploaded file from the request
    uploaded_file = request.files['csv_file']

    # Specify the column indices to fetch
    columns_to_fetch = [1, 2, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 25]

   # Read the CSV file without column names
    df = pd.read_csv(uploaded_file, usecols=columns_to_fetch, header=None,nrows=20)

    # Append two columns with all values as 0
    df['new_col1'] = 0
    df['new_col2'] = 0

    # select only the columns you want to keep
    df.columns = ['artist_familiarity', 'artist_hotttnesss', 'duration', 'end_of_fade_in', 'key',
                    'key_confidence', 'loudness', 'mode', 'mode_confidence', 'start_of_fade_out', 'tempo',
                    'time_signature', 'time_signature_confidence', 'year', 'artist_latitude', 'artist_longitude']


    # Fill empty fields with zeros
    df.fillna(0, inplace=True)

    # do some processing on the DataFrame
    outputs = []
    retrained_output = []
    # iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Extract the numeric features for the current row
        features = row.loc[numeric_features].values

        # Predict the output using the loaded model
        output = model.predict(features.reshape(1, -1))[0]
        output2 = model2.predict(features.reshape(1, -1))[0]

        # Append the output to the outputs array
        outputs.append(output)
        retrained_output.append(output2)

    # Convert the outputs list to a plain text response
    plain_text = '\n'.join(map(str, outputs))
    plain_text2 = '\n'.join(map(str, retrained_output))

    # Concatenate the heading with the plain text response
    heading = "Initial model results:<br>"
    plain_text_with_heading = heading + plain_text

    # Concatenate the heading with the plain text2 response
    heading2 = "<br> Retrained model results:<br>"
    plain_text2_with_heading = heading2 + plain_text2

    # Concatenate the plain text and plain text2 responses
    response = plain_text_with_heading + '<br><br>' + plain_text2_with_heading

    # Return the response with the heading
    return response


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


