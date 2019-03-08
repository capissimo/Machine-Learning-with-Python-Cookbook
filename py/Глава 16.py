
# coding: utf-8

# # Глава 16. 
# ## Логистическая регрессия
# > <b>16.1 Тренировка бинарного классификатора

# In[1]:


# Загрузить библиотеки
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

# Загрузить данные только с двумя классами
iris = datasets.load_iris()
features = iris.data[:100,:]
target = iris.target[:100]

# Стандартизировать признаки
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

# Создать объект логистической регрессии
logistic_regression = LogisticRegression(random_state=0)

# Натренировать модель
model = logistic_regression.fit(features_standardized, target)


# In[2]:


# Создать новое наблюдение
new_observation = [[.5, .5, .5, .5]]

# Предсказать класс
model.predict(new_observation)


# In[3]:


# Взглянуть на предсказанные вероятности
model.predict_proba(new_observation)


# > <b>16.2 Тренировка мультиклассового классификатора

# In[4]:


# Загрузить библиотеки
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

# Загрузить данные
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Стандартизировать признаки
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

# Создать объект логистической регрессии 
# по методу "один против остальных"
logistic_regression = LogisticRegression(random_state=0, multi_class="ovr")

# Натренировать модель
model = logistic_regression.fit(features_standardized, target)


# > <b>16.3 Снижение дисперсии с помощью регуляризации

# In[5]:


# Загрузить библиотеки
from sklearn.linear_model import LogisticRegressionCV
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

# Загрузить данные
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Стандартизировать признаки
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

# Создать объект-классификатор на основе дерева принятия решений
logistic_regression = LogisticRegressionCV(
    penalty='l2', Cs=10, random_state=0, n_jobs=-1)

# Натренировать модель
model = logistic_regression.fit(features_standardized, target)


# > <b>16.4 Тренировка классификатора на очень крупных данных

# In[6]:


# Загрузить библиотеки
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

# Загрузить данные
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Стандартизировать признаки
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

# Создать объект логистической регрессии
logistic_regression = LogisticRegression(random_state=0, solver="sag")

# Натренировать модель
model = logistic_regression.fit(features_standardized, target)


# > <b>16.5 Обработка несбалансированных классов

# In[7]:


# Загрузить библиотеки
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

# Загрузить данные
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Сделать класс сильно несбалансированным,
# удалив первые 40 наблюдений
features = features[40:,:]
target = target[40:]

# Создать вектор целей, указав либо класс 0, либо 1
target = np.where((target == 0), 0, 1)

# Стандартизировать признаки
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

# Создать объект-классификатор дерева принятия решений
logistic_regression = LogisticRegression(random_state=0, class_weight="balanced")

# Натренировать модель
model = logistic_regression.fit(features_standardized, target)

