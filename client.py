import requests

url = "http://127.0.0.1:8000/predict"

data = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}

response = requests.post(url, json=data)

if response.status_code == 200:
    result = response.json()
    print("Ответ от сервера:")
    print(f"Код: {result['prediction_code']}")
    print(f"Цветок: {result['flower_name']}")
else:
    print(f"Ошибка сервера: {response.status_code}")
