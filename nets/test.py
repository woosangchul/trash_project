import requests

url = 'http://127.0.0.1:8000/api/images'
files = {'file': open('C\\temp\\5.jpg", 'rb')}
r = requests.post(url, files=files)