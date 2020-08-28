import pandas as pd
from flask import Flask, jsonify, request
import pickle
from vect_tweets.Vectorizing import VectTweet

# load model
model = pickle.load(open('model.pkl','rb'))

# app
app = Flask(__name__)

# routes
@app.route('/predict', methods=['POST'])

def predict():
    # get data
    test_json = request.get_json(force=True)
    # collect data
    if test_json:
        
        if isinstance(test_json, dict): #unique value
        
            data = pd.DataFrame(test_json, index=[0])
        else:
            data = pd.DataFrame(test_json, columns = test_json[0].keys())


    # data preparation
    vect = VectTweet()
    
    df1 = vect.data_preparation(data)

    # predictions
    pred = model.predict(df1)

    data['prediction'] = pred

    return data.to_json(orient = 'records')

if __name__ == '__main__':
    app.run(port = 5000, debug=True)