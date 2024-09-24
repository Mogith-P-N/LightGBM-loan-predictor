from flask import Flask, request, jsonify
import pandas as pd

import lightgbm as lgb


def model_deployment(model_path):
    app = Flask(__name__)
    model  = lgb.Booster(model_file=model_path)
    
    @app.route('/predict', methods=['POST'])
    def predict():
        data = request.get_json(force=True)
        data = pd.DataFrame(data)
        prediction = model.predict(data)
        return jsonify(prediction.tolist())
    
    if __name__ == '__main__':
        app.run(host='0.0.0.0', port=8080, debug=True)
