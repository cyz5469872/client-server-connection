from flask import Flask
from flask_restful import Resource, Api
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


app = Flask(__name__)
api = Api(app)

class ROC(Resource):
    def get(self, preprocessing, c):
		 # you need to preprocess the data according to user preferences (only fit preprocessing on train data)
        if(preprocessing=="standard"):
            scaler = StandardScaler(with_std=False)
            scaler.fit(X_train)
            standardized_train=scaler.transform(X_train)
        if(preprocessing=="min-max"):
            scaler=MinMaxScaler()
            scaler.fit(X_train)
            standardized_train=scaler.transform(X_train)
		# fit the model on the training set
        lr=LogisticRegression(C=c)
        lr_model=lr.fit(standardized_train,y_train)
		# predict probabilities on test set
        predicted_result=lr_model.predict(X_test)
        predicted_result = predicted_result.tolist()
        fpr, tpr, thresholds = roc_curve(y_test,predicted_result)
        fpr=fpr.tolist()
        tpr=tpr.tolist()
        thresholds=thresholds.tolist()
		# return the false positives, true positives, and thresholds using roc_curve()
        return {
                'fpr':fpr,
                'tpr':tpr,
                'thresholds':thresholds
                }
# Here you need to add the ROC resource, ex: api.add_resource(HelloWorld, '/')
api.add_resource(ROC, '/<string:preprocessing>/<float:c>')
# for examples see 
# https://flask-restful.readthedocs.io/en/latest/quickstart.html#a-minimal-api

if __name__ == '__main__':
    # load data
    df = pd.read_csv('C:/Users/yzcvo/Desktop/DS5500/client connection/transfusion.data')
    xDf = df.loc[:, df.columns != 'Donated']
    y = df['Donated']
	# get random numbers to split into train and test
    np.random.seed(1)
    r = np.random.rand(len(df))
	# split into train test
    X_train = xDf[r < 0.8]
    X_test = xDf[r >= 0.8]
    y_train = y[r < 0.8]
    y_test = y[r >= 0.8]
    app.run(debug=True)
