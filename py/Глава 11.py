
# coding: utf-8

# # Глава 11. 
# ## Оценивание моделей
# > <b>11.1 Перекрестная проверка моделей

# In[2]:


# Загрузить библиотеки
from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Загрузить набор данных рукописных цифр
digits = datasets.load_digits()

# Создать матрицу признаков
features = digits.data

# Создать вектор целей
target = digits.target

# Создать стандартизатор
standardizer = StandardScaler()

# Создать объект логистической регрессии
logit = LogisticRegression()

# Создать конвейер, который стандартизирует, 
# затем выполняет логистическую регрессию
pipeline = make_pipeline(standardizer, logit)

# Создать k-блочную перекрестную проверку
kf = KFold(n_splits=10, shuffle=True, random_state=1)

# Выполнить k-блочную перекрестную проверку
cv_results = cross_val_score(pipeline,  # Конвейер
                             features,  # Матрица признаков
                             target,    # Вектор целей
                             cv=kf,     # Метод перекрестной проверки
                             scoring="accuracy", # Функция потери
                             n_jobs=-1) # Использовать все ядра CPU

# Вычислить среднее значение
cv_results.mean()


# In[3]:


# Взглянуть на оценки для всех 10 блоков
cv_results


# In[4]:


# Импортировать библиотеку
from sklearn.model_selection import train_test_split

# Создать тренировочный и тестовый наборы
features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.1, random_state=1)

# Выполнить подгонку стндартизатора к тренировочному набору
standardizer.fit(features_train)

# Применить к обоим наборам: тренировочному и тестовому
features_train_std = standardizer.transform(features_train)
features_test_std = standardizer.transform(features_test)


# In[5]:


# Создать конвейер
pipeline = make_pipeline(standardizer, logit)


# In[6]:


# Выполнить k-блочную перекрестную проверку
cv_results = cross_val_score(pipeline,  # Конвейер
                             features,  # Матрица признаков
                             target,    # Вектор целей
                             cv=kf,     # Метод перекрестной проверки
                             scoring="accuracy", # Функция потери
                             n_jobs=-1) # Использовать все ядра CPU


# > <b>11.2 Создание регрессионной модели в качестве ориентира

# In[8]:


# Загрузить библиотеки
from sklearn.datasets import load_boston
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split

# Загрузить данные
boston = load_boston()

# Создать признаки
features, target = boston.data, boston.target

# Выполнить разбивку на тренировочный и тестовый наборы
features_train, features_test, target_train, target_test =     train_test_split(features, target, random_state=0)

# Создать фиктивный регрессор
dummy = DummyRegressor(strategy='mean')

# "Натренировать" фиктивный регрессор
dummy.fit(features_train, target_train)

# Получить оценку R-квадрат
dummy.score(features_test, target_test)


# In[8]:


# Загрузить библиотеки
from sklearn.linear_model import LinearRegression

# Натренировать простую линейно-регрессионную модель
ols = LinearRegression()
ols.fit(features_train, target_train)

# Получить оценку R-квадрат
ols.score(features_test, target_test)


# In[9]:


# Создать фиктивный регрессор, который 
# предсказывает 20 для всех наблюдений
clf = DummyRegressor(strategy='constant', constant=20)
clf.fit(features_train, target_train)

# Вычислить оценку
clf.score(features_test, target_test)


# > <b>11.3 Создание классификационной модели в качестве ориентира

# In[7]:


# Загрузить библиотеки
from sklearn.datasets import load_iris
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split

# Загрузить данные
iris = load_iris()

# Создать матрицу признаков и вектор целей
features, target = iris.data, iris.target

# Разбить на тренировочный и тестовый наборы
features_train, features_test, target_train, target_test =     train_test_split(features, target, random_state=0)

# Создать фиктивный классификатор
dummy = DummyClassifier(strategy='uniform', random_state=1)

# "Натренировать" модель
dummy.fit(features_train, target_train)

# Получить оценку точности
dummy.score(features_test, target_test)


# In[11]:


# Загрузить библиотеку
from sklearn.ensemble import RandomForestClassifier

# Создать классификатор случайного леса
classifier = RandomForestClassifier()

# Натренировать модель
classifier.fit(features_train, target_train)

# Получить оценку точности
classifier.score(features_test, target_test)


# > <b>11.4 Оценивание предсказаний бинарного классификатора

# In[1]:


# Загрузить библиотеки
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Сгенерировать матрицу признаков и вектор целей
X, y = make_classification(n_samples = 10000,
                           n_features = 3,
                           n_informative = 3,
                           n_redundant = 0,
                           n_classes = 2,
                           random_state = 1)

# Создать объект логистической регрессии
logit = LogisticRegression()

# Перекрестно проверить модель, 
# используя показатель точности
cross_val_score(logit, X, y, scoring="accuracy")


# In[2]:


# Перекрестно проверить модель, 
# используя показатель прецизионности
cross_val_score(logit, X, y, scoring="precision")


# In[3]:


# Перекрестно проверить модель, 
# используя показатель полноты
cross_val_score(logit, X, y, scoring="recall")


# In[4]:


# Перекрестно проверить модель, 
# используя показатель f1
cross_val_score(logit, X, y, scoring="f1")


# In[5]:


# Загрузить библиотеку
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Разбить на тренировочный и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.1,
                                                    random_state=1)

# Предсказать значения для тренировочного вектора целей
y_hat = logit.fit(X_train, y_train).predict(X_test)

# Вычислить точность
accuracy_score(y_test, y_hat)


# > <b>11.5 Оценивание порогов бинарного классификатора

# *Определение стиля изображения*

# In[17]:


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


# In[18]:


# Загрузить библиотеки
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split

# Создать матрицу признаков и вектор целей
features, target = make_classification(n_samples=10000,
                                       n_features=10,
                                       n_classes=2,
                                       n_informative=3,
                                       random_state=3)

# Разбить на тренировочный и тестовый наборы
features_train, features_test, target_train, target_test = train_test_split(
features, target, test_size=0.1, random_state=1)

# Создать логистический регрессионный классификатор 
logit = LogisticRegression()

# Натренировать модель
logit.fit(features_train, target_train)

# Получить предсказанные вероятности
target_probabilities = logit.predict_proba(features_test)[:,1]

# Создать доли истинно- и ложноположительных исходов
false_positive_rate, true_positive_rate, threshold = roc_curve(target_test,
target_probabilities)

# Построить график кривой ROC
plt.title("Кривая ROC")
plt.plot(false_positive_rate, true_positive_rate)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel("Доля истинноположительных исходов")
plt.xlabel("Доля ложноположительных исходов")
plt.tight_layout()
plt.savefig('pics/11_01.png', dpi=600)
plt.show()


# In[19]:


# Получить предсказанные вероятности
logit.predict_proba(features_test)[0:1]


# In[20]:


logit.classes_


# In[22]:


print("Порог:", threshold[116])
print("Доля истинноположительных:", true_positive_rate[116])
print("Доля ложноположительных:", false_positive_rate[116])


# In[23]:


print("Порог:", threshold[45])
print("Доля истинноположительных:", true_positive_rate[45])
print("Доля ложноположительных:", false_positive_rate[45])


# In[24]:


# Вычислить площадь под кривой
roc_auc_score(target_test, target_probabilities)


# > <b>11.6 Оценивание предсказаний мультиклассового классификатора

# In[25]:


# Загрузить библиотеки
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Сгенерировать матрицу признаков и вектор целей
features, target = make_classification(n_samples = 10000,
                                       n_features = 3,
                                       n_informative = 3,
                                       n_redundant = 0,
                                       n_classes = 3,
                                       random_state = 1)

# Создать объект логистической регрессии
logit = LogisticRegression()

# Перекрестно проверить модель,
# используя показатель точности
cross_val_score(logit, features, target, scoring='accuracy')


# In[26]:


# Перекрестно проверить модель,
# используя макро-усредненную оценку F1
cross_val_score(logit, features, target, scoring='f1_macro')


# > <b>11.7 Визуализация результативности классификатора

# In[31]:


# Загрузить библиотеки
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd

# Загрузить данные
iris = datasets.load_iris()

# Создать матрицу признаков
features = iris.data

# Создать вектор целей
target = iris.target

# Создать список имен целевых классов
class_names = iris.target_names

# Создать тренировочный и тестовый наборы
features_train, features_test, target_train, target_test = train_test_split(
features, target, random_state=1)

# Создать объект логистической регрессии
classifier = LogisticRegression()

# Натренировать модель и сделать предсказания
target_predicted = classifier.fit(features_train,
target_train).predict(features_test)

# Создать матрицу ошибок
matrix = confusion_matrix(target_test, target_predicted)

# Создать фрейм данных pandas
dataframe = pd.DataFrame(matrix, index=class_names, columns=class_names)

# Создать тепловую картуа
sns.heatmap(dataframe, annot=True, cbar=None, cmap="Blues")
plt.title("Матрица ошибок"), plt.tight_layout()
plt.ylabel("Истинный класс"), plt.xlabel("Предсказанный класс")
plt.tight_layout()
plt.savefig('pics/11_02.png', dpi=600)
plt.show()


# > <b>11.8 Оценивание регрессионных моделей

# In[32]:


# Загрузить библиотеки
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

# Сгенерировать матрицу признаков, вектор целей
features, target = make_regression(n_samples = 100,
                                   n_features = 3,
                                   n_informative = 3,
                                   n_targets = 1,
                                   noise = 50,
                                   coef = False,
                                   random_state = 1)

# Создать объект линейной регрессии
ols = LinearRegression()

# Перекрестно проверить линейную регрессию,
# используя (отрицательный) показатель MSE
cross_val_score(ols, features, target, scoring='neg_mean_squared_error')


# In[33]:


# Перекрестно проверить линейную регрессию,
# используя показатель R-квадрат
cross_val_score(ols, features, target, scoring='r2')


# > <b>11.9 Оценивание кластеризующих моделей

# In[34]:


# Загрузить библиотеки
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Сгененировать матрицу признаков
features, _ = make_blobs(n_samples = 1000,
                         n_features = 10,
                         centers = 2,
                         cluster_std = 0.5,
                         shuffle = True,
                         random_state = 1)

# Кластеризовать данные, используя алгоритм k-средних,
# чтобы предсказать классы
model = KMeans(n_clusters=2, random_state=1).fit(features)

# Получить предсказанные классы
target_predicted = model.labels_

# Оценить модель
silhouette_score(features, target_predicted)


# > <b>11.10 Создание собственного оценочного метрического показателя

# In[35]:


# Загрузить библиотеки
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.datasets import make_regression

# Сгененировать матрицу признаков и вектор целей
features, target = make_regression(n_samples = 100,
                                   n_features = 3,
                                   random_state = 1)

# Создать тренировочный и тестовый наборы
features_train, features_test, target_train, target_test = train_test_split(
features, target, test_size=0.10, random_state=1)

# Создать собственный метрический показатель
def custom_metric(target_test, target_predicted):
    # Вычислить оценочный показатель r-квадрат
    r2 = r2_score(target_test, target_predicted)
    # Вернуть оценочный показатель r-квадрат
    return r2

# Создать оценочную функцию и установить,  
# что чем выше оценки, тем они лучше
score = make_scorer(custom_metric, greater_is_better=True)

# Создать объект гребневой регрессии
classifier = Ridge()

# Натренировать гребневую регрессионную модель
model = classifier.fit(features_train, target_train)

# Применить собственную оценочную функцию
score(model, features_test, target_test)


# In[36]:


# Предсказать значения
target_predicted = model.predict(features_test)

# Вычислить оценочный показатель r-квадрат
r2_score(target_test, target_predicted)


# > <b>11.11 Визуализация эффекта размера тренировочного набора

# In[40]:


# Загрузить библиотеки
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve

# Загрузить данные
digits = load_digits()

# Создать матрицу признаков и вектор целей
features, target = digits.data, digits.target

# Создать перекрестно-проверочные тренировочные и тестовые 
# оценки для разных размеров тренировочного набора
train_sizes, train_scores, test_scores = learning_curve(
                                    RandomForestClassifier(), # Классификатор
                                    features, # Матрица признаков
                                    target,   # Вектор целей
                                    cv=10,    # Количество блоков
                                    scoring='accuracy', # Показатель
                                                        # результативности
                                    n_jobs=-1,  # Использовать все ядра CPU
                                    # Размеры 50 тренировочных наборов
                                    train_sizes=np.linspace(0.01, 1.0, 50))

# Создать стредние и стандартные отклонения оценок 
# тренировочного набора
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

# Создать стредние и стандартные отклонения оценок 
# тестового набора
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Нанести линии
plt.plot(train_sizes, train_mean, '--', color="#111111", 
         label="Тренировочная оценка")
plt.plot(train_sizes, test_mean, color="#111111", 
         label="Перекрестно-проверочная оценка")

# Нанести полосы
plt.fill_between(train_sizes, train_mean - train_std,
train_mean + train_std, color="#DDDDDD")
plt.fill_between(train_sizes, test_mean - test_std,
test_mean + test_std, color="#DDDDDD")

# Построить график
plt.title("Кривая заучивания")
plt.xlabel("Размер тренировочного набора"), plt.ylabel("Оценка точности"),
plt.legend(loc="best")
plt.tight_layout()
plt.savefig('pics/11_03.png', dpi=600)
plt.show()


# > <b>11.12 Создание текстового отчета об оценочных метрических показателях

# In[41]:


# Загрузить библиотеки
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Загрузить данные
iris = datasets.load_iris()

# Создать матрицу признаков
features = iris.data

# Создать вектор целей
target = iris.target

# Создать список имен целевых классов
class_names = iris.target_names

# Создать тренировочны й тестовый наборы
features_train, features_test, target_train, target_test = train_test_split(
features, target, random_state=1)

# Создать объект логистической регрессии
classifier = LogisticRegression()

# Натренировать модель и сделать предсказания
model = classifier.fit(features_train, target_train)
target_predicted = model.predict(features_test)

# Создать классификационный отчет
print(classification_report(target_test,
                            target_predicted,
                            target_names=class_names))


# > <b>11.13 Визуализация эффекта значений гиперпараметра

# In[43]:


# Загрузить библиотеки
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import validation_curve

# Загрузить данные
digits = load_digits()

# Создать матрицу признаков и вектор целей
features, target = digits.data, digits.target

# Создать диапазон значений для параметра
param_range = np.arange(1, 250, 2)

# Вычислить точность на тренировочном и тестовом наборах,
# используя диапазон значений параметра
train_scores, test_scores = validation_curve(
                RandomForestClassifier(),  # Классификатор
                features,                  # Матрица признаков
                target,                    # Вектор целей
                param_name="n_estimators", # Исследуемый гиперпараметр
                param_range=param_range,   # Диапазон значений гиперпараметра
                cv=3,                      # Количество блоков
                scoring="accuracy",        # Показатель результативности
                n_jobs=-1)                 # Использовать все ядра CPU

# Вычислить среднее и стандартное отклонение для оценок
# тренировочного набора
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

# Вычислить среднее и стандартное отклонение для оценок
# тестового набора
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Построить график средних оценок точности 
# для тренировочного и тестового наборов
plt.plot(param_range, train_mean, color="black",
         label="Тренировочная оценка",)
plt.plot(param_range, test_mean, color="dimgrey",
         label="Перекрестно-проверочная оценка")

# Нанести полосы точности 
# для тренировочного и тестового наборов
plt.fill_between(param_range, train_mean - train_std,
train_mean + train_std, color="gray")
plt.fill_between(param_range, test_mean - test_std,
test_mean + test_std, color="gainsboro")

# Создать график
plt.title("Валидационная кривая со случайным лесом")
plt.xlabel("Количество деревьей")
plt.ylabel("Оценка точности")
plt.tight_layout()
plt.legend(loc="best")
plt.tight_layout()
plt.savefig('pics/11_04.png', dpi=600)
plt.show()

