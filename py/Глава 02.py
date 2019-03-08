
# coding: utf-8

# # Глава 2. 
# ## Загрузка данных
# > <b>2.1 Загрузка образца набора данных

# In[1]:


# Загрузить наборы данных scikit-learn
from sklearn import datasets

# Загрузить набор изображений рукописных цифр
digits = datasets.load_digits()

# Создать матрицу признаков
features = digits.data

# Создать вектор целей
target = digits.target

# Взглянуть на первое наблюдение 
features[0]


# > <b>2.2 Создание симулированного набора данных

# In[1]:


# Загрузить библиотеку
from sklearn.datasets import make_regression

# Сгенерировать матрицу признаков, вектор целей и истинные коэффициенты
features, target, coefficients = make_regression(n_samples = 100,
                                                 n_features = 3,
                                                 n_informative = 3,
                                                 n_targets = 1,
                                                 noise = 0.0,
                                                 coef = True,
                                                 random_state = 1)

# Взглянуть на матрицу признаков и вектор целей
print('Матрица признаков\n', features[:3])
print('Вектор целей\n', target[:3])


# In[4]:


# Загрузить библиотеку
from sklearn.datasets import make_classification

# Сгенерировать матрицу признаков и вектор целей
features, target = make_classification(n_samples = 100,
                                       n_features = 3,
                                       n_informative = 3,
                                       n_redundant = 0,
                                       n_classes = 2,
                                       weights = [.25, .75],
                                       random_state = 1)

# Взглянуть на признаковую матрицу и вектор целей
print('Матрица признаков\n', features[:3])
print('Вектор целей\n', target[:3])


# In[6]:


# Загрузить библиотеку
from sklearn.datasets import make_blobs

# Сгенерировать матрицу признаков и вектор целей
features, target = make_blobs(n_samples = 100,
                              n_features = 2,
                              centers = 3,
                              cluster_std = 0.5,
                              shuffle = True,
                              random_state = 1)

# Взглянуть на признаковую матрицу и вектор целей
print('Матрица признаков\n', features[:3])
print('Вектор целей\n', target[:3])


# *Определение стиля изображения*

# In[7]:


# Заставить все графики в блокноте Jupyter
# в дальнейшем появляться локально 
get_ipython().run_line_magic('matplotlib', 'inline')

# переопределение стиля
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family']     = 'sans-serif'
rcParams['font.sans-serif'] = ['Ubuntu Condensed']
rcParams['figure.figsize']  = (4, 3)
rcParams['legend.fontsize'] = 10
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9


# In[8]:


# Загрузить библиотеку
import matplotlib.pyplot as plt

# Взглянуть на диаграмму рассеяния
plt.scatter(features[:,0], features[:,1], c=target)
plt.tight_layout()
plt.savefig('pics/2_02.png', dpi=600)
plt.show()


# > <b>2.3 Загрузка файла CSV

# In[6]:


# Загрузить библиотеку
import pandas as pd

# Создать URL-адрес
url = 'https://tinyurl.com/simulated-data' # или локально 'data.csv'


# Загрузить набор данных
dataframe = pd.read_csv(url)

# Взглянуть на первые две строки
dataframe.head(2)


# > <b>2.4 Загрузка файла Excel

# In[7]:


# Загрузить библиотеку
import pandas as pd

# Создать URL-адрес
url = 'https://tinyurl.com/simulated-excel' # или локально 'data.xlsx'

# Загрузить данные
dataframe = pd.read_excel(url, sheet_name=0, header=1)

# Взглянуть на первые две строки
dataframe.head(2)


# > <b>2.5 Загрузка файла JSON

# In[9]:


# Загрузить библиотеку
import pandas as pd

# Создать URL-адрес
url = 'https://tinyurl.com/simulated-json' # или локально 'data.json'

# Загрузить данные
dataframe = pd.read_json(url, orient='columns')

# Взглянуть на первые две строки
dataframe.head(2)


# > <b>2.6 Опрашивание базы данных SQL

# In[13]:


# Загрузить библиотеки
import pandas as pd
from sqlalchemy import create_engine

# Создать подключение к базе данных
database_connection = create_engine('sqlite:///sample.db')

# Загрузить данные
dataframe = pd.read_sql_query('SELECT * FROM data', database_connection)

# Взглянуть на первые две строки
dataframe.head(2)

