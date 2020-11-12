# Significance Test API
from flask import Flask, request
from flask_restful import Resource, Api
from flask_cors import CORS
from flask.helpers import make_response
import statsmodels.stats.api as sms
import pandas as pd
from scipy import stats
import s3fs
import io
import json
app = Flask(__name__)
CORS(app)
api = Api(app)
class process(Resource):
    def post(self):
        body = request.get_json(silent=True) or {}
        file1 = body['file_path1']
        file2 = body['file_path2']
        feature_columns = ['msisdn', 'ad_id', 'event_type']
        data1 = pd.read_csv(file1,usecols=feature_columns)
        data2 = pd.read_csv(file2,usecols=feature_columns)
        data = pd.concat([data1, data2])
        ad_list = list(data['ad_id'].value_counts()[:2].index)
        dictionary = {ad_list[0]:0, ad_list[1]:1}
        data['ad_id'] = data['ad_id'].map(dictionary)
        data['event_type'] = data['event_type'].map({'sms':0, 'click':1, 'Click':1})
        test = stats.ttest_ind(data.loc[data['ad_id'] == 1]['event_type'],data.loc[data['ad_id'] == 0]['event_type'], equal_var=False)
        
        if (test.pvalue>0.05):
            print ("Non-significant results")
            return {
                'error' : False,
                'message' : 'Non-significant results',
                'data': test,
                'status_code' : 200
            }, 200
            
        elif (test.statistic>0):
            print ("Test group has statistically better results")
            return {
                'error' : False,
                'message' : 'Test group has statistically better results',
                'data': test,
                'status_code' : 200
            }, 200
        else:
            print ("Test group has statistically worse results")
            return {
                'error' : False,
                'message' : 'Test group has statistically worse results',
                'data': test,
                'status_code' : 200
            }, 200
api.add_resource(process, '/get_stat_significance')
if __name__ == '__main__':
    print("App running")
    app.run(host="0.0.0.0", port=5000, debug=True)