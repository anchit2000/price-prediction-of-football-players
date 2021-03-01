import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

model = pickle.load(open("/Users/NKAPUR/PycharmProjects/Football_project/model.pkl", 'rb'))

@app.route('/')
def home():
    return render_template("form.html")

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    data = {}
    data['name'] = [str(request.form.get('name'))]
    data['clubname'] = [str(request.form.get('clubname'))]
    data['age'] = [int(str(request.form.get('age')))]
    data['position'] = [str(request.form.get('position'))]
    data['position_cat'] = [int(str(request.form.get('position_cat')))]
    data['page_views'] = [int(str(request.form.get('page_views')))]
    data['fpl_value'] = [int(str(request.form.get('fpl_value')))]
    data['fpl_sel'] = [int(str(request.form.get('fpl_sel')))]
    data['fpl_points'] = [int(str(request.form.get('fpl_points')))]
    data['region'] = [int(str(request.form.get('region')))]
    data['nationality'] = [str(request.form.get('nationality'))]
    data['new_foreign'] = [int(str(request.form.get('new_foreign')))]
    data['age_cat'] = [int(str(request.form.get('age_cat')))]
    data['club_id'] = [int(str(request.form.get('club_id')))]
    data['big_club'] = [int(str(request.form.get('big_club')))]
    data['new_signing'] = [int(str(request.form.get('new_signing')))]

    data = pd.DataFrame.from_dict(data)

    final_data = data
    final_data.drop(columns=['name', 'clubname', 'position', 'nationality', 'age'], inplace=True, axis = 1)

    final_features = final_data.to_numpy()
    prediction = model.predict(final_features)
    output = prediction[0]

    return render_template('form.html', prediction_text='Price of the player should be nearly $ {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)