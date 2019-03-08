
# coding: utf-8

# # Глава 12. 
# ## Отбор модели
# > <b>12.1 Отбор наилучшей модели с помощью исчерпывающего поиска

# In[1]:


# Загрузить библиотеки
import numpy as np
from sklearn import linear_model, datasets
from sklearn.model_selection import GridSearchCV

# Загрузить данные
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Создать объект логистической регрессии
logistic = linear_model.LogisticRegression()

# Создать диапазон вариантов значений 
# штрафного гиперпараметра
penalty = ['l1', 'l2']

# Создать диапазон вариантов значений 
# регуляризационного гиперпараметра
C = np.logspace(0, 4, 10)

# Создать словарь вариантов гиперпараметров
hyperparameters = dict(C=C, penalty=penalty)

# Создать объект решеточного поиска
gridsearch = GridSearchCV(logistic, hyperparameters, cv=5, verbose=0)

# Выполнить подгонку объекта решеточного поиска
best_model = gridsearch.fit(features, target)


# In[2]:


np.logspace(0, 4, 10)


# In[3]:


# Взглянуть на наилучшие гиперпараметры
print('Лучший штраф:', best_model.best_estimator_.get_params()['penalty'])
print('Лучший C:', best_model.best_estimator_.get_params()['C'])


# In[4]:


# Предсказать вектор целей
best_model.predict(features)


# > <b>12.2 Отбор наилучших моделей с помощью рандомизированного поиска

# In[5]:


# Загрузить библиотеки
from scipy.stats import uniform
from sklearn import linear_model, datasets
from sklearn.model_selection import RandomizedSearchCV

# Загрузить данные
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Создать объект логистической регрессии
logistic = linear_model.LogisticRegression()

# Создать диапазон вариантов значений 
# штрафного гиперпараметра
penalty = ['l1', 'l2']

# Создать диапазон вариантов значений 
# регуляризационного гиперпараметра
C = uniform(loc=0, scale=4)

# Создать словарь вариантов гиперпараметров
hyperparameters = dict(C=C, penalty=penalty)

# Создать объект рандомизированного поиска
randomizedsearch = RandomizedSearchCV(
    logistic, hyperparameters, random_state=1, n_iter=100, cv=5, verbose=0,
    n_jobs=-1)

# Выполнить подгонку объекта рандомизированного поиска
best_model = randomizedsearch.fit(features, target)


# In[6]:


# Определить равномерное распределение между 0 и 4, 
# отобрать 10 значений
uniform(loc=0, scale=4).rvs(10)


# In[7]:


# Взглянуть на самые лучшие гиперпараметры
print('Лучший штраф:', best_model.best_estimator_.get_params()['penalty'])
print('Лучший C:', best_model.best_estimator_.get_params()['C'])


# In[8]:


# Предсказать вектор целей
best_model.predict(features)


# > <b>12.3 Отбор наилучших моделей из нескольких обучающихся алгоритмов

# In[9]:


# Загрузить библиотеки
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# Задать начальное число для генератора псевдослучайных чисел
np.random.seed(0)

# Згрузить данные
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Создать конвейер
pipe = Pipeline([("classifier", RandomForestClassifier())])

# Создать словарь вариантов обучающихся алгоритмов и их гиперпараметров
search_space = [{"classifier": [LogisticRegression()],
                 "classifier__penalty": ['l1', 'l2'],
                 "classifier__C": np.logspace(0, 4, 10)},
                {"classifier": [RandomForestClassifier()],
                 "classifier__n_estimators": [10, 100, 1000],
                 "classifier__max_features": [1, 2, 3]}]

# Создать объект решеточного поиска
gridsearch = GridSearchCV(pipe, search_space, cv=5, verbose=0)

# Выполнить подгонку объекта решеточного поиска
best_model = gridsearch.fit(features, target)


# In[10]:


# Взгялнуть на самую лучшую модель
best_model.best_estimator_.get_params()["classifier"]


# In[11]:


# Предсказать вектор целей
best_model.predict(features)


# > <b>12.4 Отбор наилучших моделей во время предобработки

# In[12]:


# Загрузить библиотеки
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Задать начальное число для генератора случайных чисел
np.random.seed(0)

# Згрузить данные 
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Создать объект предобработки, который включает
# признаки стандартного шкалировщика StandardScaler и объект PCA
preprocess = FeatureUnion([("std", StandardScaler()), ("pca", PCA())])

# Создать конвейер
pipe = Pipeline([("preprocess", preprocess),
("classifier", LogisticRegression())])

# Создать пространство вариантов значений
search_space = [{"preprocess__pca__n_components": [1, 2, 3],
                 "classifier__penalty": ["l1", "l2"],
                 "classifier__C": np.logspace(0, 4, 10)}]

# Создать объект решеточного поиска
clf = GridSearchCV(pipe, search_space, cv=5, verbose=0, n_jobs=-1)

# Выполнить подгонку объекта решеточного поиска
best_model = clf.fit(features, target)


# In[13]:


# Взглянуть на самую лучшую модель
best_model.best_estimator_.get_params()['preprocess__pca__n_components']


# > <b>12.5 Ускорение отбора модели с помощию распараллеливания

# In[14]:


# Загрузить библиотеки
import numpy as np
from sklearn import linear_model, datasets
from sklearn.model_selection import GridSearchCV

# Загрузить данные
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Создать объект логистической регрессии
logistic = linear_model.LogisticRegression()

# Создать диапазон вариантов значений 
# штрафного гиперпараметра
penalty = ["l1", "l2"]

# Создать диапазон вариантов значений 
# регуляризационного гиперпараметра
C = np.logspace(0, 4, 1000)

# Создать словарь вариантов гиперпараметров
hyperparameters = dict(C=C, penalty=penalty)

# Создать объект решеточного поиска
gridsearch = GridSearchCV(logistic, hyperparameters, cv=5, n_jobs=-1, verbose=1)

# Выполнить подгонку объекта решеточного поиска
best_model = gridsearch.fit(features, target)


# In[15]:


# Создать объект решеточного поиска с использованием одного ядра
clf = GridSearchCV(logistic, hyperparameters, cv=5, n_jobs=1, verbose=1)

# Выполнить подгонку объекта решеточного поиска
best_model = clf.fit(features, target)


# > <b>12.6 Ускорение отбора модели с помощью алгоритмически специализированных методов

# In[16]:


# Загрузить библиотеку
from sklearn import linear_model, datasets

# Загрузить данные
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Создать объект перекрестно-проверяемой логистической регрессии
logit = linear_model.LogisticRegressionCV(Cs=100)

# Натренировать модель
logit.fit(features, target)


# > <b>12.7 Оценивание результативности после отбора модели

# In[17]:


# Загрузить библиотеки
import numpy as np
from sklearn import linear_model, datasets
from sklearn.model_selection import GridSearchCV, cross_val_score

# Загрузить данные
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Создать объект логистической регрессии
logistic = linear_model.LogisticRegression()

# Создать диапазон из 20 вариантов значений для C
C = np.logspace(0, 4, 20)

# Создать словарь вариантов гиперпараметров
hyperparameters = dict(C=C)

# Создать объект решеточного поиска
gridsearch = GridSearchCV(logistic, hyperparameters, cv=5, n_jobs=-1, verbose=0)

# Выполнить вложенную перекрестную проверку и выдать среднюю оценку
cross_val_score(gridsearch, features, target).mean()


# In[18]:


gridsearch = GridSearchCV(logistic, hyperparameters, cv=5, verbose=1)


# In[19]:


best_model = gridsearch.fit(features, target)


# In[20]:


scores = cross_val_score(gridsearch, features, target)

