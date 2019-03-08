
# coding: utf-8

# # Глава 19. 
# ## Кластеризация
# > <b>19.1 Кластеризация с помощью k-средних

# In[1]:


# Загрузить библиотеки
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Загрузить данные
iris = datasets.load_iris()
features = iris.data

# Стандартизировать признаки
scaler = StandardScaler()
features_std = scaler.fit_transform(features)

# Создать объект k-средних
cluster = KMeans(n_clusters=3, random_state=0, n_jobs=-1)

# Натренировать модель
model = cluster.fit(features_std)


# In[2]:


# Взглянуть на предсказанный класс
model.labels_


# In[3]:


# Взглянуть на истинный класс
iris.target


# In[4]:


# Создать новое наблюдение
new_observation = [[0.8, 0.8, 0.8, 0.8]]

# Предсказать кластер наблюдения
model.predict(new_observation)


# In[5]:


# Взглянуть на центры кластеров
model.cluster_centers_


# > <b>19.2 Ускорение кластеризации методом k-средних

# In[6]:


# Загрузить библиотеки
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans

# Загрузить данные
iris = datasets.load_iris()
features = iris.data

# Стандартизировать признаки
scaler = StandardScaler()
features_std = scaler.fit_transform(features)

# Создать объект k-средних
cluster = MiniBatchKMeans(n_clusters=3, random_state=0, batch_size=100)

# Натренировать модель
model = cluster.fit(features_std)


# > <b>19.3 Кластеризация методом сдвига к среднему

# In[8]:


# Загрузить библиотеки
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MeanShift

# Загрузить данные
iris = datasets.load_iris()
features = iris.data

# Стандартизировать признаки
scaler = StandardScaler()
features_std = scaler.fit_transform(features)

# Создать объект кластеризации методом сдвига к среднему
cluster = MeanShift(n_jobs=-1)

# Натренировать модель
model = cluster.fit(features_std)


# In[9]:


# Показать принадлежность к кластерам
model.labels_


# > <b>19.4 Кластеризация методом DBSCAN

# In[10]:


# Загрузить библиотеки
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# Загрузить данные
iris = datasets.load_iris()
features = iris.data

# Стандартизировать признаки
scaler = StandardScaler()
features_std = scaler.fit_transform(features)

# Создать объект плотностной кластеризации dbscan
cluster = DBSCAN(n_jobs=-1)

# Натренировать модель
model = cluster.fit(features_std)


# In[11]:


# Показать принадлежность к кластерам
model.labels_


# > <b>19.5 Кластеризация методом иерархического слияния

# In[13]:


# Загрузить библиотеки
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

# Загрузить данные
iris = datasets.load_iris()
features = iris.data

# Стандартизировать признаки
scaler = StandardScaler()
features_std = scaler.fit_transform(features)

# Создать объект агломеративной кластеризации
cluster = AgglomerativeClustering(n_clusters=3)

# Натренировать модель
model = cluster.fit(features_std)


# In[14]:


model.labels_

