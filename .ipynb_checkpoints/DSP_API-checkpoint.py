#The code below is the flask app on the server.
from __future__ import print_function
import post_process
import json
import pickle
import preprocess_data
import numpy as np
import json
import request
import flask
import pandas as pd
import io


model_path = "/home/ec2-user/linear_model/models/pickle_model.pkl"
model_in = open(model_path, "rb")
model = pickle.load(model_in)

app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_model(model_path) is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    IDs = []
    FEATURE = ['id','device_type','os_vendor','os_name','os_version','browser_name','browser_version','location_region','location_state','app_name','site_name','site_link','exchange','carrier','captured_time',
 'device_screen_height','device_screen_width','device_screen_pixel_ratio']
    data = None
    file =  flask.request.files["data"]
    if file.content_type == 'text/csv':
        data = flask.request.files["data"]
        data = pd.read_csv(data,usecols=FEATURE)
        for msisdn in list(data.id):
            IDs.append(msisdn)
        data['captured_time'] = data['captured_time'].astype(str)
    elif file.content_type == 'application/json':
        data = flask.request.files["data"]
        data = pd.read_json(data)
        data=data[FEATURE]
        for msisdn in list(data.id):
            IDs.append(msisdn)    
        data['captured_time'] = data['captured_time'].astype(str)
    else:
        return flask.Response(response='This predictor only supports CSV/JSON data', status=414, mimetype='text/plain')

    dataframe = post_process.pipeline_object(data)
    predictions = model.predict_proba(dataframe)[:,1]
    out = io.StringIO()
    pd.DataFrame({'results':predictions.flatten()}).to_csv(out, index=False,sep=',',header=['score'])
    result = out.getvalue()
    return result

@app.route('/raw_requests', methods=['POST'])
def raw_request():
    IDs = []
    FEATURE = ['id','device_type','os_vendor','os_name','os_version','browser_name','browser_version','location_region','location_state','app_name','site_name','site_link','exchange','carrier','captured_time',
 'device_screen_height','device_screen_width','device_screen_pixel_ratio']
    df = None
    df = flask.request.get_json(force=True)
    data = pd.io.json.json_normalize(df)
    data=data[FEATURE]
    for msisdn in list(data['id']):
        IDs.append(msisdn) 
    data['captured_time'] = data['captured_time'].astype(str)
    dataframe = post_process.pipeline_object(data)
    predictions = model.predict_proba(dataframe)[:,1]
    out = io.StringIO()
    pd.DataFrame({'results':predictions.flatten()}).to_csv(out, index=False,sep=',',header=['score'])
    result = out.getvalue()
    return result

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7000, debug=True)
    
    