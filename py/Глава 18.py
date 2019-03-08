
# coding: utf-8

# # Глава 18. 
# ## Наивный Байес
# > <b>18.1 Тренировка классификатора для непрерывных признаков

# In[2]:


# Загрузить библиотеки
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB

# Загрузить данные
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Создать объект гаусова наивного Байеса
classifer = GaussianNB()

# Натренировать модель
model = classifer.fit(features, target)


# In[3]:


# Создать новое наблюдение
new_observation = [[ 4, 4, 4, 0.4]]

# Предсказать класс
model.predict(new_observation)


# In[4]:


# Создать объект гаусова наивного Байеса 
# с априорными вероятностями для каждого класса
clf = GaussianNB(priors=[0.25, 0.25, 0.5])

# Натренировать модель
model = classifer.fit(features, target)


# > <b>18.2 Тренировка классификатора для дискертных и счетных признаков

# In[5]:


# Загрузить библиотеки
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# Создать текст
text_data = np.array(['Бразилия - моя любовь. Бразилия!',
                      'Бразилия - лучше',
                      'Германия бьет обоих'])

# Создать мешок слов
count = CountVectorizer()
bag_of_words = count.fit_transform(text_data)

# Создать матрицу признаков
features = bag_of_words.toarray()

# Создать вектор целей
target = np.array([0,0,1])

# Создать объект полиномиального наивного байесова классификатора
# с априорными вероятностями каждого класса
classifer = MultinomialNB(class_prior=[0.25, 0.5])

# Натренировать модель
model = classifer.fit(features, target)


# In[6]:


# Создать новое наблюдение
new_observation = [[0, 0, 0, 1, 0, 1, 0]]

# Предсказать класс нового наблюдения
model.predict(new_observation)


# > <b>18.3 Тренировка наивного байесова классификатора для бинарных признаков

# In[7]:


# Загрузить библиотеки
import numpy as np
from sklearn.naive_bayes import BernoulliNB

# Создать три бинарных признака
features = np.random.randint(2, size=(100, 3))

# Создать вектор бинарных целей
target = np.random.randint(2, size=(100, 1)).ravel()

# Создать объект бернуллиева наивного Байеса
# с априорными вероятностями каждого класса
classifer = BernoulliNB(class_prior=[0.25, 0.5])

# Натренировать модель
model = classifer.fit(features, target)


# In[8]:


model_uniform_prior = BernoulliNB(class_prior=None, fit_prior=True)


# > <b>18.4 Калибровка предсказанных вероятностей

# In[9]:


# Загрузить библиотеки
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV

# Загрузить данные
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Создать объект гауссова наивного Байеса
classifer = GaussianNB()

# Создать откалиброванную перекрестную проверку
# с сигмоидальной калибровкой
classifer_sigmoid = CalibratedClassifierCV(classifer, cv=2, method='sigmoid')

# Откалибровать вероятности
classifer_sigmoid.fit(features, target)

# Создать новое наблюдение
new_observation = [[ 2.6, 2.6, 2.6, 0.4]]

# Взглянуть на откалиброванные вероятности
classifer_sigmoid.predict_proba(new_observation)


# In[10]:


# Натренировать гауссов наивный Байес и
# затем предсказать вероятности классов
classifer.fit(features, target).predict_proba(new_observation)


# In[11]:


# Взглянуть на откалиброванные вероятности
classifer_sigmoid.predict_proba(new_observation)

