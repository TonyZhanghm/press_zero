import requests

url = 'http://localhost:5000/predict'

prompt = "Hi! How are you?"
data = {'input': prompt+"<|respond|>"}

response = requests.post(url, json=data)

predictions = response.json()['predictions']

print(predictions)