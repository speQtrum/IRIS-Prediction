from flask import Flask, request
from flask_restful import Resource, Api
import numpy as np
import pickle



app = Flask(__name__)
model = pickle.load(open('webapp_103/model.pkl', 'rb'))
app.config['SECRET_KEY'] = 'mysecretkey'
api = Api(app)



class Iris_Prediction(Resource):
    def post(self):  # adds a student to database
        data = request.get_json()
        input_features = [data["sepal length"],data["sepal width"],
                              data["petal length"],data["petal width"]]
        final_features = [np.array(input_features)]
        prediction = model.predict(final_features)[0]

        return {"message":"Prediction successful","class":prediction}


api.add_resource(Iris_Prediction,'/')


if __name__ == '__main__':
    app.run(debug = True)