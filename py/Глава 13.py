
# coding: utf-8

# # Глава 13. 
# ## Линейная регрессия
# > <b>13.1 Подгонка прямой

# In[1]:


# Загрузить библиотеки
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston

# Загрузить данные только с двумя признаками
boston = load_boston()
features = boston.data[:,0:2]
target = boston.target

# Создать объект линейной регрессии
regression = LinearRegression()

# Выполнить подгонку линейной регрессии
model = regression.fit(features, target)


# In[2]:


# Взглянуть на точку пересечения
model.intercept_


# In[3]:


# Взглянуть на коэффициенты признаков
model.coef_


# In[4]:


# Первое значение в векторе целей, умноженное на 1000
target[0]*1000


# In[5]:


# Предсказать целевое значение первого наблюдения, умноженное на 1000
model.predict(features)[0]*1000


# In[6]:


# Первый коэффициент, умноженный на 1000
model.coef_[0]*1000


# > <b>13.2 Обработка интерактивных эффектов

# In[7]:


# Загрузить библиотеки
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.preprocessing import PolynomialFeatures

# Загрузить данные только с двумя признаками
boston = load_boston()
features = boston.data[:,0:2]
target = boston.target

# Создать член, характеризующий взаимодействие между признаками
interaction = PolynomialFeatures(
    degree=3, include_bias=False, interaction_only=True)
features_interaction = interaction.fit_transform(features)

# Создать объект линейной регрессии
regression = LinearRegression()

# Выполнить подгонку линейной регрессии
model = regression.fit(features_interaction, target)


# In[8]:


# Взглянуть на признаки для первого наблюдения
features[0]


# In[9]:


# Импортировать библиотеку
import numpy as np

# Для каждого наблюдения умножить значения первого и второго признака
interaction_term = np.multiply(features[:, 0], features[:, 1])


# In[10]:


# Взглянуть на член взаимодействия для первого наблюдения
interaction_term[0]


# In[11]:


# Взглянуть на значения для первого наблюдения
features_interaction[0]


# > <b>13.3 Подгонка нелинейной связи

# In[12]:


# Загрузить библиотеки
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.preprocessing import PolynomialFeatures

# Загрузить данные с одним признаком
boston = load_boston()
features = boston.data[:,0:1]
target = boston.target

# Создать полиномиальные признаки x^2 и x^3
polynomial = PolynomialFeatures(degree=3, include_bias=False)
features_polynomial = polynomial.fit_transform(features)

# Создать объект линейной регрессии
regression = LinearRegression()

# Выполнить подогонку линейной регрессии
model = regression.fit(features_polynomial, target)


# In[13]:


# Взглянуть на первое наблюдение
features[0]


# In[14]:


# Взглянуть на первое наблюдение, возведенное во вторую степень, x^2
features[0]**2


# > <b>13.4 Снижение дисперсии с помощью регуляризации

# In[15]:


# Загрузить библиотеки
from sklearn.linear_model import Ridge
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler

# Загрузить данные
boston = load_boston()
features = boston.data
target = boston.target

# Стандартизировать признаки
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

# Созздать объект гребневой регрессии со значением альфа
regression = Ridge(alpha=0.5)

# Выполнить подгонку линейной регрессии
model = regression.fit(features_standardized, target)


# In[16]:


# Загрузить библиотеку
from sklearn.linear_model import RidgeCV

# Создать объект гребневой регрессии с тремя значениями alpha
regr_cv = RidgeCV(alphas=[0.1, 1.0, 10.0])

# Выполнить подгонку линейной регрессии
model_cv = regr_cv.fit(features_standardized, target)

# Взглянуть на коэффициенты
model_cv.coef_


# In[17]:


# Взглянуть на alpha
model_cv.alpha_


# > <b>13.5 Уменьшение количества признаков с помощью лассо-регрессии

# In[18]:


# Загрузить библиотеки
from sklearn.linear_model import Lasso
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler

# Загрузить данные
boston = load_boston()
features = boston.data
target = boston.target

# Стандартизировать признаки
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

# Создать объект ласс-регрессии со значением alpha
regression = Lasso(alpha=0.5)

# Выполнить подгонку линейной регрессии
model = regression.fit(features_standardized, target)


# In[19]:


# Взглянуть на коэффициенты
model.coef_


# In[20]:


# Создать лассо-регрессию с высоким alpha
regression_a10 = Lasso(alpha=10)
model_a10 = regression_a10.fit(features_standardized, target)
model_a10.coef_

