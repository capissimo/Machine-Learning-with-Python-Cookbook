
# coding: utf-8

# # Глава 14. 
# ## Деревья и леса
# > <b>14.1 Тренировка классификационного дерева принятия решений

# In[1]:


# Загрузить библиотеки
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets

# Загрузить данные
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Создать объект-классификатор дерева принятия решений
decisiontree = DecisionTreeClassifier(random_state=0)

# Натренировать модель
model = decisiontree.fit(features, target)


# In[2]:


# Сконструировать новое наблюдение
observation = [[ 5, 4, 3, 2]]

# Предсказать класс наблюдения
model.predict(observation)


# In[3]:


# Взглянуть на предсказанные вероятности трех классов
model.predict_proba(observation)


# In[4]:


# Создать объект-классификатор дерева принятия решений, 
# используя энтропию
decisiontree_entropy = DecisionTreeClassifier(
    criterion='entropy', random_state=0)

# Натренировать модель
model_entropy = decisiontree_entropy.fit(features, target)


# > <b>14.2 Тренировка регрессионного дерева принятия решений

# In[7]:


# Загрузить библиотеки
from sklearn.tree import DecisionTreeRegressor
from sklearn import datasets

# Загрузить данные всего с двумя признаками
boston = datasets.load_boston()
features = boston.data[:,0:2]
target = boston.target

# Создать объект-классификатор дерева принятия решений
decisiontree = DecisionTreeRegressor(random_state=0)

# Натренировать модель
model = decisiontree.fit(features, target)


# In[8]:


# Сконструировать новое наблюдение
observation = [[0.02, 16]]

# Предсказать значение наблюдения
model.predict(observation)


# In[9]:


# Создать объект-классификатор дерева принятия решений,
# используя энтропию
decisiontree_mae = DecisionTreeRegressor(criterion="mae", random_state=0)

# Натренировать модель
model_mae = decisiontree_mae.fit(features, target)


# > <b>14.3 Визуализация модели дерева принятия решений

# In[12]:


# Загрузить библиотеки
import pydotplus
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from IPython.display import Image
from sklearn import tree

# Загрузить данные
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Создать объект-классификатор дерева принятия решений
decisiontree = DecisionTreeClassifier(random_state=0)

# Натренировать модель
model = decisiontree.fit(features, target)

# Создать данные в формате DOT
dot_data = tree.export_graphviz(decisiontree,
                                out_file=None,
                                feature_names=iris.feature_names,
                                class_names=iris.target_names)

# Начертить граф
graph = pydotplus.graph_from_dot_data(dot_data)

# Показать граф
Image(graph.create_png())


# In[13]:


# Создать PDF
graph.write_pdf("iris.pdf")
True

# Создать PNG
graph.write_png("iris.png")
True


# > <b>14.4 Тренировка классификационного случайного леса

# In[14]:


# Загрузить библиотеки
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets

# Загрузить данные
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Создать объект-классификатор случайного леса
randomforest = RandomForestClassifier(random_state=0, n_jobs=-1)

# Натренировать модель
model = randomforest.fit(features, target)


# In[15]:


# Сконструировать новое наблюдение
observation = [[ 5, 4, 3, 2]]

# Предсказать класс наблюдения
model.predict(observation)


# In[17]:


# Создать объект-классификатор случайного леса,
# используя энтропию
randomforest_entropy = RandomForestClassifier(
    criterion="entropy", random_state=0)

# Натренировать модель
model_entropy = randomforest_entropy.fit(features, target)


# > <b>14.5 Тренировка регрессионного случайного леса

# In[18]:


# Загрузить библиотеки
from sklearn.ensemble import RandomForestRegressor
from sklearn import datasets

# Загрузить данные только с двумя признаками
boston = datasets.load_boston()
features = boston.data[:,0:2]
target = boston.target

# Создать объект-классификатор случайного леса
randomforest = RandomForestRegressor(random_state=0, n_jobs=-1)

# Натренировать модель
model = randomforest.fit(features, target)


# > <b>14.6 Идентификация важных признаков в случайных лесах

# In[20]:


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


# In[21]:


# Загрузить библиотеки
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets

# Загрузить данные
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Создать объект-классификатор случайного леса
randomforest = RandomForestClassifier(random_state=0, n_jobs=-1)

# Натренировать модель
model = randomforest.fit(features, target)

# Вычислить важности признаков
importances = model.feature_importances_

# Отсортировать важности признаков в нисходящем порядке
indices = np.argsort(importances)[::-1]

# Перераспределить имена признаков, чтобы они совпадали 
# с отсортированными важностями признаков
names = [iris.feature_names[i] for i in indices]

# Создать график
plt.figure()

# Создать заголовок графика
plt.title("Важности признаков")

# Добавить столбики
plt.bar(range(features.shape[1]), importances[indices])

# Добавить имена признаков как метки оси X
plt.xticks(range(features.shape[1]), names, rotation=90)

# Показать график
plt.tight_layout()
plt.savefig('pics/14_06.png', dpi=600)
plt.show()


# In[22]:


# Взглянуть на важности признаков
model.feature_importances_


# > <b>14.7 Отбор важных признаков в случайных лесах

# In[23]:


# Загрузить библиотеки
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.feature_selection import SelectFromModel

# Загрузить данные
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Создать объект-классификатор случайного леса
randomforest = RandomForestClassifier(random_state=0, n_jobs=-1)

# Создать объект, который отбирает признаки с важностью,
# большей или равной порогу
selector = SelectFromModel(randomforest, threshold=0.3)

# Выполнить подгонку новой матрицы признаков, используя селектор
features_important = selector.fit_transform(features, target)

# Натренировать случаный лес, используя наиболее важные признаки
model = randomforest.fit(features_important, target)


# > <b>14.8 Обработка несбалансированных классов

# In[24]:


# Загрузить библиотеки
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets

# Загрузить данные
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Сделать класс сильно несбалансированным,
# удалив первые 40 наблюдений
features = features[40:,:]
target = target[40:]

# Создать вектор целей, назначив классам значения 0 либо 1
target = np.where((target == 0), 0, 1)

# Создать объект-классификатор случайного леса
randomforest = RandomForestClassifier(
    random_state=0, n_jobs=-1, class_weight="balanced")

# Натренировать модель
model = randomforest.fit(features, target)


# In[25]:


# Вычислить вес для малого класса
110/(2*10)


# In[26]:


# Вычислить вес для крупного класса
110/(2*100)


# > <b>14.9 Управление размером дерева

# In[27]:


# Загрузить библиотеки
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets

# Загрузить данные
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Создать объект-классификатор дерева принятия решений
decisiontree = DecisionTreeClassifier(random_state=0,
                                      max_depth=None,
                                      min_samples_split=2,
                                      min_samples_leaf=1,
                                      min_weight_fraction_leaf=0,
                                      max_leaf_nodes=None,
                                      min_impurity_decrease=0)

# Натренировать модель
model = decisiontree.fit(features, target)


# > <b>14.10 Улучшение результативности с помощью бустинга

# In[28]:


# Загрузить библиотеки
from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets

# Загрузить данные
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Создать объект-классификатор дерева принятия решений 
# на основе алгоритма adaboost
adaboost = AdaBoostClassifier(random_state=0)

# Натренировать модель
model = adaboost.fit(features, target)


# > <b>14.11 Оценивание случайных лесов с помощью ошибок внепакетных наблюдений

# In[29]:


# Загрузить библиотеки
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets

# Загрузить данные
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Создать объект-классификатор случайного леса
randomforest = RandomForestClassifier(
    random_state=0, n_estimators=1000, oob_score=True, n_jobs=-1)

# Натренировать модель
model = randomforest.fit(features, target)

# Взглянуть на внепакетную ошибку
randomforest.oob_score_

