
# coding: utf-8

# # Глава 15. 
# ## K-ближайших соседей
# > <b>15.1 Отыскание ближайших соседей наблюдения

# In[1]:


# Загрузить библиотеки
from sklearn import datasets
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Загрузить данные
iris = datasets.load_iris()
features = iris.data

# Создать стандартизатор
standardizer = StandardScaler()

# Стандартизировать признаки
features_standardized = standardizer.fit_transform(features)

# Два ближайших соседа
nearest_neighbors = NearestNeighbors(n_neighbors=2).fit(features_standardized)

# Создать наблюдение
new_observation = [ 1, 1, 1, 1]

# Найти расстояния и индексы ближайших соседей наблюдения
distances, indices = nearest_neighbors.kneighbors([new_observation])

# Взглянуть на ближайших соседей
features_standardized[indices]


# In[2]:


# Взглянуть на расстояния
distances


# In[3]:


# Найти трех ближайших соседей каждого наблюдения
# на основе евклидова расстояния (включая себя)
nearestneighbors_euclidean = NearestNeighbors(
    n_neighbors=3, metric="euclidean").fit(features_standardized)

# Список списков, показывающий 3 ближайших соседей 
# каждого наблюдения (включая себя)
nearest_neighbors_with_self = nearestneighbors_euclidean.kneighbors_graph(
    features_standardized).toarray()

# Удалить единицы, отметив наблюдение, как ближайший сосед к себе
for i, x in enumerate(nearest_neighbors_with_self):
    x[i] = 0

# Взглянуть на ближайших соседей первого наблюдения
nearest_neighbors_with_self[0]


# > <b>15.2 Создание классификационной модели k-ближайших соседей

# In[4]:


# Загрузить библиотеки
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import datasets

# Загрузить данные
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Создать стандартизатор
standardizer = StandardScaler()

# Стандартизировать признаки
X_std = standardizer.fit_transform(X)

# Натренировать классификатор KNN с 5 соседями
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1).fit(X_std, y)

# Создать два наблюдения
new_observations = [[ 0.75, 0.75, 0.75, 0.75],
                    [ 1, 1, 1, 1]]

# Предсказать класс двух наблюдений
knn.predict(new_observations)


# In[5]:


# Взглянуть на вероятность, что каждое наблюдения 
# является одним из трех классов
knn.predict_proba(new_observations)


# In[6]:


knn.predict(new_observations)


# > <b>15.3 Идентификация наилучшего размера окрестности

# In[7]:


# Загрузить библиотеки
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV

# Загрузить данные
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Создать стандартизатор
standardizer = StandardScaler()

# Стандартизовать признаки
features_standardized = standardizer.fit_transform(features)

# Создать классификатор KNN
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)

# Создать конвейер
pipe = Pipeline([("standardizer", standardizer), ("knn", knn)])

# Создать пространство вариантов значений
search_space = [{"knn__n_neighbors": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}]

# Создать объект решеточного поиска
classifier = GridSearchCV(
    pipe, search_space, cv=5, verbose=0).fit(features_standardized, target)


# In[8]:


# Наилучший размер окрестности (k)
classifier.best_estimator_.get_params()["knn__n_neighbors"]


# > <b>15.4 Создание радиусного классификатора ближайших соседей

# In[9]:


# Загрузить библиотеки
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import datasets

# Загрузить данные
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Создать стандартизатор
standardizer = StandardScaler()

# Стандартизовать признаки
features_standardized = standardizer.fit_transform(features)

# Натренировать радиусный классификатор соседей
rnn = RadiusNeighborsClassifier(
    radius=.5, n_jobs=-1).fit(features_standardized, target)

# Создать наблюдение
new_observations = [[ 1, 1, 1, 1]]

# Предсказать класс наблюдения
rnn.predict(new_observations)


# In[10]:


# Создать два наблюдения
new_observations = [[ 0.75, 0.75, 0.75, 0.75],
                    [ 1, 1, 1, 1]]

# Предсказать класс двух наблюдений
rnn.predict(new_observations)

