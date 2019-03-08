
# coding: utf-8

# # Глава 5. 
# ## Работа с категориальными данными
# > <b>5.1 Кодирование номинальных категориальных признаков

# In[1]:


# Импортировать библиотеки
import numpy as np
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer

# Создать признак
feature = np.array([["Texas"],
                    ["California"],
                    ["Texas"],
                    ["Delaware"],
                    ["Texas"]])

# Создать кодировщик одного активного состояния
one_hot = LabelBinarizer()

# Преобразовать признак 
# в кодировку с одним активным состоянием
one_hot.fit_transform(feature)


# In[2]:


# Взглянуть на классы признака
one_hot.classes_


# In[3]:


# Обратить кодирование с одним активным состоянием
one_hot.inverse_transform(one_hot.transform(feature))


# In[4]:


# Импортировать библиотеку
import pandas as pd

# Создать фиктивные переменные из признака
pd.get_dummies(feature[:,0])


# In[5]:


# Создать мультиклассовый признак
multiclass_feature = [("Texas", "Florida"),
                      ("California", "Alabama"),
                      ("Texas", "Florida"),
                      ("Delware", "Florida"),
                      ("Texas", "Alabama")]

# Создать мультиклассовый кодировщик, преобразующий 
# в кодировку с одним активным состоянием
one_hot_multiclass = MultiLabelBinarizer()

# Преобразовать мультиклассовый признак 
# в кодировку с одним активным состоянием
one_hot_multiclass.fit_transform(multiclass_feature)


# In[6]:


# Взглянуть на классы
one_hot_multiclass.classes_


# > <b>5.2 Кодирование порядковых категориальных признаков

# In[28]:


# Загрузить библиотеку
import pandas as pd

# Создать признаки
dataframe = pd.DataFrame({"оценка": ["низкая", "низкая", 
                                     "средняя", "средняя", "высокая"]})

# Создать словарь преобразования шкалы 
scale_mapper = {"низкая":1,
                "средняя":2,
                "высокая":3}

# Заменить значения признаков значениями словаря
dataframe["оценка"].replace(scale_mapper)


# In[29]:


dataframe = pd.DataFrame({"оценка": ["низкая",
                                     "низкая",
                                     "средняя",
                                     "средняя",
                                     "высокая",
                                     "чуть больше средней"]})
scale_mapper = {"низкая":1,
                "средняя":2,
                "чуть больше средней": 3,
                "высокая":4}

dataframe["оценка"].replace(scale_mapper)


# In[27]:


scale_mapper = {"низкая":1,
                "средняя":2,
                "чуть больше средней": 2.1,
                "высокая":3}

dataframe["оценка"].replace(scale_mapper)


# > <b>5.3 Кодирование словарей признаков

# In[21]:


# Импортировать библиотеку
from sklearn.feature_extraction import DictVectorizer

# Создать словарь
data_dict = [{"красный": 2, "синий": 4},
             {"красный": 4, "синий": 3},
             {"красный": 1, "желтый": 2},
             {"красный": 2, "желтый": 2}]

# Создать векторизатор словаря
dictvectorizer = DictVectorizer(sparse=False)

# Конвертировать словарь в матрицу признаков
features = dictvectorizer.fit_transform(data_dict)

# Взглянуть на матрицу признаков
features


# In[22]:


# Получить имена признаков
feature_names = dictvectorizer.get_feature_names()

# Взглянуть на имена признаков
feature_names


# In[23]:


# Импортировать библиотеку
import pandas as pd

# Создать фрейм данных из признаков
pd.DataFrame(features, columns=feature_names)


# In[20]:


# Создать словари частотностей слов для четырех документов
doc_1_word_count = {"красный": 2, "синий": 4}
doc_2_word_count = {"красный": 4, "синий": 3}
doc_3_word_count = {"красный": 1, "желтый": 2}
doc_4_word_count = {"красный": 2, "желтый": 2}

# Создать список
doc_word_counts = [doc_1_word_count,
                   doc_2_word_count,
                   doc_3_word_count,
                   doc_4_word_count]

# Конвертировать список словарей частотностей слов в матрицу признаков
dictvectorizer.fit_transform(doc_word_counts)


# > <b>5.4 Импутация пропущенных значений классов

# In[30]:


# Загрузить библиотеки
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Создать матрицу признаков с категориальным признаком
X = np.array([[0, 2.10, 1.45],
              [1, 1.18, 1.33],
              [0, 1.22, 1.27],
              [1, -0.21, -1.19]])

# Создать матрицу признаков с отсутстующими значениями в категориальном признаке
X_with_nan = np.array([[np.nan, 0.87, 1.31],
                       [np.nan, -0.67, -0.22]])

# Натренировать ученика KNN
clf = KNeighborsClassifier(3, weights='distance')
trained_model = clf.fit(X[:,1:], X[:,0])

# Предсказать класс пропущенных значений
imputed_values = trained_model.predict(X_with_nan[:,1:])

# Соединить столбец предсказанного класса с их другими признаками
X_with_imputed = np.hstack((imputed_values.reshape(-1,1), X_with_nan[:,1:]))

# Соединить две матрицы признаков
np.vstack((X_with_imputed, X))


# In[31]:


from sklearn.preprocessing import Imputer

# Соединить две матрицы признаков
X_complete = np.vstack((X_with_nan, X))
imputer = Imputer(strategy='most_frequent', axis=0)
imputer.fit_transform(X_complete)


# > <b>5.5 Работа с несбалансированными классами

# In[36]:


# Загрузить библиотеки
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Загрузить данные цветков ириса Фишера
iris = load_iris()

# Создать матрицу признаков
features = iris.data

# Создать вектор целей
target = iris.target

# Удалить первые 40 наблюдений
features = features[40:,:]
target = target[40:]

# Создать бинарный вектор целей, указывающий, является ли класс 0
target = np.where((target == 0), 0, 1)

# Взглянуть на несбалансированный вектор целей
target


# In[38]:


# Создать веса
weights = {0: .9, 1: 0.1}

# Создать классификатор на основе случайного леса с весами
RandomForestClassifier(class_weight=weights)
RandomForestClassifier(bootstrap=True, class_weight={0: 0.9, 1: 0.1},
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, min_impurity_decrease=0.0,
                       min_impurity_split=None, min_samples_leaf=1,
                       min_samples_split=2, min_weight_fraction_leaf=0.0,
                       n_estimators=10, n_jobs=1, oob_score=False, 
                       random_state=None, verbose=0, warm_start=False)


# In[34]:


# Натренировать случайный лес с помощью сбалансированных весов классов
RandomForestClassifier(class_weight="balanced")

RandomForestClassifier(bootstrap=True, class_weight='balanced',
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, min_impurity_decrease=0.0,
                       min_impurity_split=None, min_samples_leaf=1,
                       min_samples_split=2, min_weight_fraction_leaf=0.0,
                       n_estimators=10, n_jobs=1, oob_score=False, 
                       random_state=None, verbose=0, warm_start=False)


# In[40]:


# Индексы наблюдений каждого класса
i_class0 = np.where(target == 0)[0]
i_class1 = np.where(target == 1)[0]

# Количество наблюдений в каждом классе
n_class0 = len(i_class0)
n_class1 = len(i_class1)

# Для каждого наблюдения класса 0 сделать случайную выборку
# из класса 1 без возврата
i_class1_downsampled = np.random.choice(i_class1, size=n_class0, replace=False)

# Соединить вектор целей класса 0 с
# вектором целей пониженного класса 1  
np.hstack((target[i_class0], target[i_class1_downsampled]))


# In[41]:


# Соединить матрицу признаков класса 0 с
# матрицей признаков пониженного класса 1
np.vstack((features[i_class0,:], features[i_class1_downsampled,:]))[0:5]


# In[42]:


# Для каждого наблюдения в классе 1, сделать случайную выборку 
# из класса 0 с возвратом
i_class0_upsampled = np.random.choice(i_class0, size=n_class1, replace=True)

# Соединить повышенный вектор целей класса 0 с
# вектором целей класса 1
np.concatenate((target[i_class0_upsampled], target[i_class1]))


# In[43]:


# Соединить повышенную матрицу признаков класса 0 с
# матрицей признаков класса 1
np.vstack((features[i_class0_upsampled,:], features[i_class1,:]))[0:5]

