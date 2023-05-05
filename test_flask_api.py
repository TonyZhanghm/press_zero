import requests

url = 'http://localhost:5000/predict'
data = {'input': "test test"}

response = requests.post(url, json=data)

predictions = response.json()['predictions']

print(predictions)