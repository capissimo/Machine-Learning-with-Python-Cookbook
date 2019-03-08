
# coding: utf-8

# # Глава 4. 
# ## Работа с числовыми данными
# > <b>4.1 Шкалирование признака

# In[1]:


# Загрузить библиотеки 
import numpy as np
from sklearn import preprocessing

# Создать признак
feature = np.array([[-500.5],
                    [-100.1],
                    [0],
                    [100.1],
                    [900.9]])

# Создать шкалировщик
minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))

# Прошкалировать признак
scaled_feature = minmax_scale.fit_transform(feature)

# Показать прошкалированный признак
scaled_feature


# > <b>4.2 Стандартизация признака

# In[2]:


# Загрузить библиотеки
import numpy as np
from sklearn import preprocessing

# Создать признак
x = np.array([[-1000.1],
              [-200.2],
              [500.5],
              [600.6],
              [9000.9]])

# Создать шкалировщик
scaler = preprocessing.StandardScaler()

# Преобразовать признак
standardized = scaler.fit_transform(x)

# Показать признак
standardized


# In[3]:


# Напечатать среднее значение и стандартное отклонение
print("Среднее:", round(standardized.mean()))
print("Стандартное отклонение:", standardized.std())


# In[4]:


# Создать шкалировщик
robust_scaler = preprocessing.RobustScaler()

# Преобразовать признак
robust_scaler.fit_transform(x)


# > <b>4.3 Нормализация наблюдений

# In[5]:


# Загрузить библиотеки
import numpy as np
from sklearn.preprocessing import Normalizer

# Создать матрицу признаков
features = np.array([[0.5, 0.5],
                     [1.1, 3.4],
                     [1.5, 20.2],
                     [1.63, 34.4],
                     [10.9, 3.3]])

# Создать нормализатор
normalizer = Normalizer(norm="l2")

# Преобразовать матрицу признаков
normalizer.transform(features)


# In[6]:


# Преобразовать матрицу признаков
features_l2_norm = Normalizer(norm="l2").transform(features)

# Показать матрицу признаков
features_l2_norm


# In[7]:


# Преобразовать матрицу признаков
features_l1_norm = Normalizer(norm="l1").transform(features)

# Показать матрицу признаков
features_l1_norm


# In[8]:


# Напечатать сумму
print("Сумма значений первого наблюдения:",
      features_l1_norm[0, 0] + features_l1_norm[0, 1])


# > <b>4.4 Генерирование полиномиальных и взаимодействующих признаков

# In[1]:


# Загрузить библиотеки
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

# Создать матрицу признаков
features = np.array([[2, 3],
                     [2, 3],
                     [2, 3]])

# Создать объект PolynomialFeatures 
polynomial_interaction = PolynomialFeatures(degree=2, include_bias=False)

# Создать полиномиальные признаки
polynomial_interaction.fit_transform(features)


# In[2]:


interaction = PolynomialFeatures(degree=2,
                                 interaction_only=True, include_bias=False)
interaction.fit_transform(features)


# > <b>4.5 Преобразование признаков

# In[12]:


# Загрузить библиотеки 
import numpy as np
from sklearn.preprocessing import FunctionTransformer

# Создать матрицу признаков
features = np.array([[2, 3],
                     [2, 3],
                     [2, 3]])

# Определить простую функцию
def add_ten(x):
    return x + 10

# Создать проеобразователь
ten_transformer = FunctionTransformer(add_ten)

# Преобразовать матрицу признаков
ten_transformer.transform(features)


# In[13]:


# Загрузить библиотеку
import pandas as pd

# Создать фрейм данных
df = pd.DataFrame(features, columns=["признак_1", "признак_2"])

# Применить функцию
df.apply(add_ten)


# > <b>4.6 Обнаружение выбросов

# In[5]:


# Загрузить библиотеки
import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.datasets import make_blobs

# Создать симулированные данные 
features, _ = make_blobs(n_samples = 10,
                         n_features = 2,
                         centers = 1,
                         random_state = 1)

# Заменить значения первого наблюдения предельными значениями
features[0,0] = 10000
features[0,1] = 10000

# Создать детектор
outlier_detector = EllipticEnvelope(contamination=.1)

# Выполнить подгонку детектора
outlier_detector.fit(features)

# Предсказать выбросы
outlier_detector.predict(features)


# In[6]:


# Создать один признак
feature = features[:,0]

# Создать функцию, которая возращает индекс выбросов
def indicies_of_outliers(x):
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (iqr * 1.5)
    upper_bound = q3 + (iqr * 1.5)
    return np.where((x > upper_bound) | (x < lower_bound))

# Выполнить функцию
indicies_of_outliers(feature)


# > <b>4.7 Обработка выбросов

# In[9]:


# Загрузить библиотеку 
import pandas as pd

# Создать фрейм данных 
houses = pd.DataFrame()
houses['Цена'] = [534433, 392333, 293222, 4322032]
houses['Ванные'] = [2, 3.5, 2, 116]
houses['Кв_футы'] = [1500, 2500, 1500, 48000]

# Отфильтровать наблюдения
houses[houses['Ванные'] < 20]


# In[10]:


# Загрузить библиотеку 
import numpy as np

# Создать признак на основе булева условия
houses["Выброс"] = np.where(houses["Ванные"] < 20, 0, 1)

# Показать данные
houses


# In[11]:


# Взять логарифм признака
houses["Логарифм_кв_футов"] = [np.log(x) for x in houses["Кв_футы"]]

# Показать данные
houses


# > <b>4.8 Дискретизация признаков

# In[10]:


# Загрузить библиотеки
import numpy as np
from sklearn.preprocessing import Binarizer

# Создать признак
age = np.array([[6],
                [12],
                [20],
                [36],
                [65]])

# Создать бинаризатор
binarizer = Binarizer(18)

# Преобразовать признак
binarizer.fit_transform(age)


# In[11]:


# Разнести признак по корзинам
np.digitize(age, bins=[20,30,64])


# In[12]:


# Разнести признак по корзинам
np.digitize(age, bins=[20,30,64], right=True)


# In[13]:


# Разнести признак по корзинам
np.digitize(age, bins=[18])


# > <b>4.9 Группирование наблюдений с помощью кластеризации

# In[6]:


# Загрузить библиотеки
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Создать матрицу симулированных признаков
features, _ = make_blobs(n_samples = 50,
                         n_features = 2,
                         centers = 3,
                         random_state = 1)

# Создать фрейм данных
dataframe = pd.DataFrame(features, columns=["признак_1", "признак_2"])

# Создать кластеризатор по методу k-средних
clusterer = KMeans(3, random_state=0)

# Выполнить подгонку кластеризатора
clusterer.fit(features)

# Предсказать значения
dataframe["группа"] = clusterer.predict(features)

# Взглянуть на первые несколько наблюдений
dataframe.head(5)


# > <b>4.10 Удаление наблюдений с пропущенными значениями

# In[7]:


# Загрузить библиотеку
import numpy as np

# Создать матрицу признаков
features = np.array([[1.1, 11.1],
                     [2.2, 22.2],
                     [3.3, 33.3],
                     [4.4, 44.4],
                     [np.nan, 55]])

# Отставить только те наблюдения, которые не (помечены ~) пропущены
features[~np.isnan(features).any(axis=1)]


# In[8]:


# Загрузить библиотеку
import pandas as pd

# Загрузить данные
dataframe = pd.DataFrame(features, columns=["признак_1", "признак_2"])

# Удалить наблюдения с отсутствующими значениями
dataframe.dropna()


# > <b>4.11 Импутация пропущенных значений

# In[25]:


#
# Поскольку инсталляции библиотеки fancyimpute в Windows 10 вызывает
# проблемы, ниже приведен рабочий исходный код ядра этой библиотеки 
#
# Данная ячейка блокнота содержит ядро библиотеки fancyimpute
# (См. https://github.com/iskandr/fancyimpute/tree/master/fancyimpute)
# с классом KNN, содержащим метод k-ближайших соседей, 
# используемым для импутации пропущенных значений в рецепте 4.11.
#

####### Начало вставки

import numpy as np
from six.moves import range
from knnimpute import knn_impute_few_observed, knn_impute_with_argpartition

class Solver(object):
    def __init__(
            self,
            fill_method="zero",
            n_imputations=1,
            min_value=None,
            max_value=None,
            normalizer=None):
        self.fill_method = fill_method
        self.n_imputations = n_imputations
        self.min_value = min_value
        self.max_value = max_value
        self.normalizer = normalizer

    def __repr__(self):
        return str(self)

    def __str__(self):
        field_list = []
        for (k, v) in sorted(self.__dict__.items()):
            if v is None or isinstance(v, (float, int)):
                field_list.append("%s=%s" % (k, v))
            elif isinstance(v, str):
                field_list.append("%s='%s'" % (k, v))
        return "%s(%s)" % (
            self.__class__.__name__,
            ", ".join(field_list))

    def _check_input(self, X):
        if len(X.shape) != 2:
            raise ValueError("Expected 2d matrix, got %s array" % (X.shape,))

    def _check_missing_value_mask(self, missing):
        if not missing.any():
            raise ValueError("Input matrix is not missing any values")
        if missing.all():
            raise ValueError("Input matrix must have some non-missing values")

    def _fill_columns_with_fn(self, X, missing_mask, col_fn):
        for col_idx in range(X.shape[1]):
            missing_col = missing_mask[:, col_idx]
            n_missing = missing_col.sum()
            if n_missing == 0:
                continue
            col_data = X[:, col_idx]
            fill_values = col_fn(col_data)
            X[missing_col, col_idx] = fill_values

    def fill(
            self,
            X,
            missing_mask,
            fill_method=None,
            inplace=False):
        """
        Parameters
        ----------
        X : np.array
            Data array containing NaN entries
        missing_mask : np.array
            Boolean array indicating where NaN entries are
        fill_method : str
            "zero": fill missing entries with zeros
            "mean": fill with column means
            "median" : fill with column medians
            "min": fill with min value per column
            "random": fill with gaussian samples according to mean/std of column
        inplace : bool
            Modify matrix or fill a copy
        """
        if not inplace:
            X = X.copy()

        if not fill_method:
            fill_method = self.fill_method

        if fill_method not in ("zero", "mean", "median", "min", "random"):
            raise ValueError("Invalid fill method: '%s'" % (fill_method))
        elif fill_method == "zero":
            # replace NaN's with 0
            X[missing_mask] = 0
        elif fill_method == "mean":
            self._fill_columns_with_fn(X, missing_mask, np.nanmean)
        elif fill_method == "median":
            self._fill_columns_with_fn(X, missing_mask, np.nanmedian)
        elif fill_method == "min":
            self._fill_columns_with_fn(X, missing_mask, np.nanmin)
        elif fill_method == "random":
            self._fill_columns_with_fn(
                X,
                missing_mask,
                col_fn=generate_random_column_samples)
        return X

    def prepare_input_data(self, X):
        """
        Check to make sure that the input matrix and its mask of missing
        values are valid. Returns X and missing mask.
        """
        X = np.asarray(X)
        if X.dtype != "f" and X.dtype != "d":
            X = X.astype(float)

        self._check_input(X)
        missing_mask = np.isnan(X)
        self._check_missing_value_mask(missing_mask)
        return X, missing_mask

    def clip(self, X):
        """
        Clip values to fall within any global or column-wise min/max constraints
        """
        X = np.asarray(X)
        if self.min_value is not None:
            X[X < self.min_value] = self.min_value
        if self.max_value is not None:
            X[X > self.max_value] = self.max_value
        return X

    def project_result(self, X):
        """
        First undo normaliztion and then clip to the user-specified min/max
        range.
        """
        X = np.asarray(X)
        if self.normalizer is not None:
            X = self.normalizer.inverse_transform(X)
        return self.clip(X)

    def solve(self, X, missing_mask):
        """
        Given an initialized matrix X and a mask of where its missing values
        had been, return a completion of X.
        """
        raise ValueError("%s.solve not yet implemented!" % (
            self.__class__.__name__,))

    def single_imputation(self, X):
        X_original, missing_mask = self.prepare_input_data(X)
        observed_mask = ~missing_mask
        X = X_original.copy()
        if self.normalizer is not None:
            X = self.normalizer.fit_transform(X)
        X_filled = self.fill(X, missing_mask, inplace=True)
        if not isinstance(X_filled, np.ndarray):
            raise TypeError(
                "Expected %s.fill() to return NumPy array but got %s" % (
                    self.__class__.__name__,
                    type(X_filled)))

        X_result = self.solve(X_filled, missing_mask)
        if not isinstance(X_result, np.ndarray):
            raise TypeError(
                "Expected %s.solve() to return NumPy array but got %s" % (
                    self.__class__.__name__,
                    type(X_result)))

        X_result = self.project_result(X=X_result)
        X_result[observed_mask] = X_original[observed_mask]
        return X_result

    def multiple_imputations(self, X):
        """
        Generate multiple imputations of the same incomplete matrix
        """
        return [self.single_imputation(X) for _ in range(self.n_imputations)]

    def complete(self, X):
        """
        Expects 2d float matrix with NaN entries signifying missing values
        Returns completed matrix without any NaNs.
        """
        imputations = self.multiple_imputations(X)
        if len(imputations) == 1:
            return imputations[0]
        else:
            return np.mean(imputations, axis=0)

class KNN(Solver):
    """
    k-Nearest Neighbors imputation for arrays with missing data.
    Works only on dense arrays with at most a few thousand rows.
    Assumes that each feature has been centered and rescaled to have
    mean 0 and variance 1.
    
    Inspired by the implementation of kNNImpute from the R package
    imputation.
    See here: 
    https://www.rdocumentation.org/packages/imputation/versions/2.0.3/topics/kNNImpute
    """
    def __init__(
            self,
            k=5,
            orientation="rows",
            use_argpartition=False,
            print_interval=100,
            min_value=None,
            max_value=None,
            normalizer=None,
            verbose=True):
        """
        Parameters
        ----------
        k : int
            Number of neighboring rows to use for imputation.
        orientation : str
            Which axis of the input matrix should be treated as a sample
            (default is "rows" but can also be "columns")
        use_argpartition : bool
           Use a more naive implementation of kNN imputation whichs calls
           numpy.argpartition for each row/column pair. May give NaN if fewer
           than k neighbors are available for a missing value.
        print_interval : int
        min_value : float
            Minimum possible imputed value
        max_value : float
            Maximum possible imputed value
        normalizer : object
            Any object (such as BiScaler) with fit() and transform() methods
        verbose : bool
        """
        Solver.__init__(
            self,
            min_value=min_value,
            max_value=max_value,
            normalizer=normalizer)
        self.k = k
        self.verbose = verbose
        self.orientation = orientation
        self.print_interval = print_interval
        if use_argpartition:
            self._impute_fn = knn_impute_with_argpartition
        else:
            self._impute_fn = knn_impute_few_observed

    def solve(self, X, missing_mask):
        if self.orientation == "columns":
            X = X.T
            missing_mask = missing_mask.T

        elif self.orientation != "rows":
            raise ValueError(
                "Orientation must be either 'rows' or 'columns', got: %s" % (
                    self.orientation,))

        X_imputed = self._impute_fn(
            X=X,
            missing_mask=missing_mask,
            k=self.k,
            verbose=self.verbose,
            print_interval=self.print_interval)

        failed_to_impute = np.isnan(X_imputed)
        n_missing_after_imputation = failed_to_impute.sum()
        if n_missing_after_imputation != 0:
            if self.verbose:
                print("[KNN] Warning: %d/%d still missing after imputation, replacing with 0" % (
                    n_missing_after_imputation,
                    X.shape[0] * X.shape[1]))
            X_imputed[failed_to_impute] = X[failed_to_impute]

        if self.orientation == "columns":
            X_imputed = X_imputed.T

        return X_imputed
####### Конец вставки


# In[27]:


# Загрузить библиотеки
import numpy as np
# from fancyimpute import KNN     # в Win10 проблематично
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

# Создать матрицу симулированных признаков
features, _ = make_blobs(n_samples = 1000,
                         n_features = 2,
                         random_state = 1)

# Стандартизировать признаки
scaler = StandardScaler()
standardized_features = scaler.fit_transform(features)

# Заменить первое значение первого признака на пропущенное значение
true_value = standardized_features[0,0]
standardized_features[0,0] = np.nan

# Предсказать пропущенные значения в матрице признаков
# (Класс KNN из ядра библиотеки fancyimpute выше!)
features_knn_imputed = KNN(k=5, verbose=0).complete(standardized_features) 

# Сравнить истинные и импутированные значения
print("Истинное значение:", true_value)
print("Импутированное значение:", features_knn_imputed[0,0])


# In[28]:


# Загрузить библиотеку
from sklearn.preprocessing import Imputer

# Создать заполнитель 
mean_imputer = Imputer(strategy="mean", axis=0)

# Импутировать значения
features_mean_imputed = mean_imputer.fit_transform(features)

# Сравнить истинные и импутированные значения
print("Истинное значение:", true_value)
print("Импутированное значение:", features_mean_imputed[0,0])

