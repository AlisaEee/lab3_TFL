import requests

url = 'http://localhost:8081/getTests'
data = {'n_tests':5}

response = requests.post(url, json=data)
print(response.text)