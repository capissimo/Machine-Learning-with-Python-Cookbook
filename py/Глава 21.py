
# coding: utf-8

# # Глава 21. 
# ## Сохранение и загрузка натренированных моделей
# > <b>21.1 Сохранение и загрузка модели scikit-learn

# In[1]:


# Загрузить библиотеки
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.externals import joblib

# Загрузить данные
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Создать объект-классификатор случайных лесов
classifer = RandomForestClassifier()

# Натренировать модель
model = classifer.fit(features, target)

# Сохранить модель в качестве файла консервации
joblib.dump(model, "model.pkl")


# In[2]:


# Загрузить модель из файла
classifer = joblib.load("model.pkl")


# In[3]:


# Создать новое наблюдение
new_observation = [[ 5.2, 3.2, 1.1, 0.1]]

# Предсказать класс наблюдения
classifer.predict(new_observation)


# In[4]:


# Импортировать библиотеку
import sklearn

# Получить версию библиотеки scikit-learn
scikit_version = joblib.__version__

# Сохранить модель в качестве файла консервации
joblib.dump(model, "model_{version}.pkl".format(version=scikit_version))


# > <b>21.2 Сохранение и загрузка модели Keras

# In[5]:


# Загрузить библиотеки
import numpy as np
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers
from keras.models import load_model

# Задать начальное значение для ГПСЧ
np.random.seed(0)

# Задать желаемое количество признаков
number_of_features = 1000

# Загрузить данные и вектор целей из данных с отзывами о кинофильмах
(train_data, train_target), (test_data, test_target) = imdb.load_data(
    num_words=number_of_features)

# Конвертировать данные отзывов о кинофильмах в
# матрицу признаков в кодировке с одним активным состоянием
tokenizer = Tokenizer(num_words=number_of_features)
train_features = tokenizer.sequences_to_matrix(train_data, mode="binary")
test_features = tokenizer.sequences_to_matrix(test_data, mode="binary")

# Инициализировать нейронную сеть
network = models.Sequential()

# Добавить полносвязный слой с активационной функцией ReLU
network.add(layers.Dense(
    units=16,
    activation="relu",
    input_shape=(number_of_features,)))

# Добавить полносвязный слой с сигмоидальной активационной функцией
network.add(layers.Dense(units=1, activation="sigmoid"))

# Скомпилировать нейронную сеть
network.compile(
    loss="binary_crossentropy", # Перекрестная энтропия
    optimizer="rmsprop",  # Распространение СКО
    metrics=["accuracy"]) # Точностный показатель результативности

# Train neural network
history = network.fit(
    train_features, # Признаки
    train_target,   # Вектор целей
    epochs=3,       # Количество эпох
    verbose=0,      # Вывода нет
    batch_size=100, # Количество наблюдений на пакет
    validation_data=(test_features, test_target)) # Тестовые данные

# Сохранить нейронную сеть
network.save("model.h5")


# In[6]:


# Загрузить нейронную сеть
network = load_model("model.h5")

