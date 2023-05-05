from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Load your machine learning model here
    # Get the input data from the request
    # Use your model to make predictions
    # Return the predictions as a JSON object
    
    # get input data
    input_data = request.get_json()["input"]
    return jsonify({'predictions': "here's the input {} and I don't know the answer".format(input_data)})

if __name__ == '__main__':
    app.run(port=5000, debug=True)