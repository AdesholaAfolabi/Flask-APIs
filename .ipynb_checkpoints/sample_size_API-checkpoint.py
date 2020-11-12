# Sample size API
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
        sig_level = 0.05
        power = 0.8
        body = request.get_json(silent=True) or {}
        p1 = body['p1']
        p2 = body['p2']
        p1_and_p2 = sms.proportion_effectsize(p1, p2)
        sample_size = sms.NormalIndPower().solve_power(p1_and_p2, power=power, alpha=sig_level)
        return {
                'error' : False,
                'message' : 'The required sample size for this campaign is:',
                'data': round(sample_size),
                'status_code' : 200
            }, 200

api.add_resource(process, '/get_sample_size')
if __name__ == '__main__':
    print("App running")
    app.run(host="0.0.0.0", port=4000, debug=True)