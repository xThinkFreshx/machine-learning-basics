# --- 1. ИМПОРТ БИБЛИОТЕК ---
# Импортируем класс FastAPI, чтобы создать наше веб-приложение.
from fastapi import FastAPI
# Импортируем BaseModel из Pydantic. Это инструмент для строгой проверки данных, которые присылает клиент.
from pydantic import BaseModel
# Импортируем NumPy (хотя в этом коде он напрямую не используется, это хорошая практика для ML-проектов).
import numpy as np

# Импортируем наш датасет Ирисов и алгоритм Дерева решений.
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier


# --- 2. ПОДГОТОВКА МОДЕЛИ ---
# Загружаем данные ирисов в память.
iris = load_iris()

# Создаем объект модели (Дерево решений) с фиксированным random_state для воспроизводимости.
model = DecisionTreeClassifier(random_state=42)
# Обучаем модель прямо при запуске сервера! 
# Передаем ВСЕ данные (iris.data) и ВСЕ ответы (iris.target), 
# чтобы к моменту прихода запроса от клиента модель уже была готова предсказывать.
model.fit(iris.data, iris.target)


# --- 3. СОЗДАНИЕ СЕРВЕРА ---
# Создаем экземпляр приложения. Переменная "app" будет использоваться Uvicorn'ом для запуска.
# Параметр title задает название, которое ты видел в заголовке Swagger UI.
app = FastAPI(title="Iris ML API")


# --- 4. ОПИСАНИЕ ФОРМАТА ДАННЫХ (Схема) ---
# Класс FlowerFeatures наследуется от BaseModel. 
# Это "контракт" для клиента: "Я жду 4 числа, все типа float (дробные)".
# Если клиент пришлет строку вместо числа, FastAPI автоматически вернет ошибку 422 (Unprocessable Entity) 
# и НЕ пропустит запрос дальше. Это защита от дурака на уровне сети.
class FlowerFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# --- 5. ЭНДПОИНТ (Маршрут) ---
# Декоратор @app.post("/predict") означает: "Когда придет POST-запрос на адрес /predict, вызови эту функцию".
# features: FlowerFeatures — это указание серверу, что данные должны соответствовать нашей схеме выше.
@app.post("/predict")
def predict_flower(features: FlowerFeatures):
    # Формируем список из 2D-списка (как того требует sklearn при предсказании одного примера).
    # Берем данные из объекта features (который FastAPI собрал из JSON-запроса).
    data = [[features.sepal_length, features.sepal_width, features.petal_length, features.petal_width]]
    
    # Вызываем метод predict. Он возвращает массив, берем первый элемент [0] (это наш код вида: 0, 1 или 2).
    prediction = model.predict(data)[0]
    
    # Превращаем цифру (например, 0) в понятное имя ("setosa") через массив target_names.
    flower_name = iris.target_names[prediction]
    
    # Возвращаем словарь (FastAPI автоматически превратит его в JSON).
    # int(prediction) нужен, потому что sklearn возвращает специальный тип numpy.int64, 
    # который FastAPI не всегда может сериализовать в JSON без явного преобразования.
    return{
        "prediction_code": int(prediction),
        "flower_name": flower_name,
        "status": "success"
    }