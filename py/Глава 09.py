
# coding: utf-8

# # Глава 9. 
# ## Снижение размерности с помощью выделения признаков
# > <b>9.1 Снижение признаков с помощью главных компонент

# In[1]:


# Загрузить библиотеки
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import datasets

# Загрузить данные
digits = datasets.load_digits()

# Стандартизировать матрицу признаков
features = StandardScaler().fit_transform(digits.data)

# Соззать объект PCA, который сохранит 99% дисперсии
pca = PCA(n_components=0.99, whiten=True)

# Выполнить анализ PCA
features_pca = pca.fit_transform(features)

# Показать результаты
print("Исходное количество признаков:", features.shape[1])
print("Сокращенное количество признаков:", features_pca.shape[1])


# > <b>9.2 Уменьшение количества признаков, когда данные линейно неразделимы

# In[2]:


# Загрузить библиотеки
from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets import make_circles

# Создать линейно неразделимые данные
features, _ = make_circles(n_samples=1000, random_state=1, 
                           noise=0.1, factor=0.1)

# Применить ядерный PCA 
# с радиально-базисным функциональным ядром (RBF-ядром)
kpca = KernelPCA(kernel="rbf", gamma=15, n_components=1)
features_kpca = kpca.fit_transform(features)

print("Исходное количество признаков:", features.shape[1])
print("Сокращенное количество признаков:", features_kpca.shape[1])


# > <b>9.3 Уменьшение количества признаков путем максимизации разделимости классов

# In[3]:


# Загрузить библиотеки
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Загрузить набор данных цветков ириса:
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Создать объект и выполнить LDA, затем использовать
# его для преобразования признаков
lda = LinearDiscriminantAnalysis(n_components=1)
features_lda = lda.fit(features, target).transform(features)

# Напечатать количество признаков
print("Исходное количество признаков:", features.shape[1])
print("Сокращенное количество признаков:", features_lda.shape[1])


# In[4]:


lda.explained_variance_ratio_


# In[6]:


# Создать объект и выполнить LDA
lda = LinearDiscriminantAnalysis(n_components=None)
features_lda = lda.fit(features, target)

# Создать массив коэффициентов объясненной дисперсии
lda_var_ratios = lda.explained_variance_ratio_

# Создать функцию
def select_n_components(var_ratio, goal_var: float) -> int:
    # Задать исходную обясненную на данный момент дисперсию
    total_variance = 0.0

    # Задать исходное количество признаков
    n_components = 0

    # Для объясненной дисперсии каждого признака:
    for explained_variance in var_ratio:

        # Добавить объясненную дисперсию к итогу
        total_variance += explained_variance

        # Добавить единицу к количеству компонент
        n_components += 1

        # Если достигнут целевой уровень объясненной дисперсии
        if total_variance >= goal_var:
            # Завершить цикл
            break

    # Вернуть количество компонент
    return n_components

# Выполнить функцию
select_n_components(lda_var_ratios, 0.95)


# > <b>9.4 Уменьшение количества признаков с использованием разложения матрицы

# In[7]:


# Загрузить библиотеки
from sklearn.decomposition import NMF
from sklearn import datasets

# Загрузить данные
digits = datasets.load_digits()

# Загрузить матрицу признаков
features = digits.data

# Создать, выполнить подгонку и применить NMF
nmf = NMF(n_components=10, random_state=1)
features_nmf = nmf.fit_transform(features)

# Показать результаты
print("Исходное количество признаков:", features.shape[1])
print("Сокращенное количество признаков:", features_nmf.shape[1])


# > <b>9.5 Уменьшение количества признаков на разряженных данных

# In[8]:


# Загрузить библиотеки
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from sklearn import datasets
import numpy as np

# Загрузить данные
digits = datasets.load_digits()

# Стандартизировать матрицу признаков
features = StandardScaler().fit_transform(digits.data)

# Сделать разряженную матрицу 
features_sparse = csr_matrix(features)

# Создать объект TSVD
tsvd = TruncatedSVD(n_components=10)

# Выполнить TSVD на разряженной матрице
features_sparse_tsvd = tsvd.fit(features_sparse).transform(features_sparse)

# Показать результаты
print("Исходное количество признаков:", features_sparse.shape[1])
print("Сокращенное количество признаков:", features_sparse_tsvd.shape[1])


# In[9]:


# Суммировать коэффициенты объясненной дисперсии первых трех компонент
tsvd.explained_variance_ratio_[0:3].sum()


# In[10]:


# Создать и выполнить TSVD с числом признаков меньше на единицу
tsvd = TruncatedSVD(n_components=features_sparse.shape[1]-1)
features_tsvd = tsvd.fit(features)

# Поместиь в список объясненные дисперсии
tsvd_var_ratios = tsvd.explained_variance_ratio_

# Создать функцию
def select_n_components(var_ratio, goal_var):
    # Задать исходную обясненную на данный момент дисперсию
    total_variance = 0.0

    # Задать исходное количество признаков
    n_components = 0

    # Для объясненной дисперсии каждого признака:
    for explained_variance in var_ratio:

        # Добавить объясненную дисперсию к итогу
        total_variance += explained_variance

        # Добавить единицу к количеству компонент
        n_components += 1

        # Если достигнут целевой уровень объясненной дисперсии
        if total_variance >= goal_var:
            # Завершить цикл
            break

    # Вернуть количество компонент
    return n_components

# Выполнить функцию
select_n_components(tsvd_var_ratios, 0.95)

