
# coding: utf-8

# # Глава 3. 
# ## Упорядочение данных
# > <b>3.0 Введение

# In[16]:


# Загрузить библиотеку
import pandas as pd

# Создать URL-адрес
url = 'https://tinyurl.com/titanic-csv' # локально 'titanic.csv'

# Загрузить данные как фрейм данных
dataframe = pd.read_csv(url)

# Показать первые 5 строк
dataframe.head(5)


# > <b>3.1 Создание фрейма данных

# In[4]:


# Загрузить библиотеку
import pandas as pd

# Создать фрейм данных DataFrame
dataframe = pd.DataFrame()

# Добавть столбцы
dataframe['Имя'] = ['Джеки Джексон', 'Стивен Стивенсон']
dataframe['Возраст'] = [38, 25]
dataframe['Водитель'] = [True, False]

# Показать DataFrame
dataframe


# In[5]:


# Создать строку
new_person = pd.Series(['Молли Муни', 40, True], 
                       index=['Имя','Возраст','Водитель'])

# Добавить строку в конец фрейма данных
dataframe.append(new_person, ignore_index=True)


# > <b>3.2 Описание данных

# In[17]:


# Загрузить библиотеку
import pandas as pd

# Создать URL-адрес
url = 'https://tinyurl.com/titanic-csv' 

# Загрузить данные
dataframe = pd.read_csv(url)

# Показать две строки
dataframe.head(2)


# In[18]:


# Показать размерности
dataframe.shape


# In[19]:


# Показать статистику
dataframe.describe()


# > <b>3.3 Навигация по фреймам данных

# In[21]:


# Загрузить библиотеку
import pandas as pd

# Создать URL-адрес
url = 'https://tinyurl.com/titanic-csv' 

# Загрузить данные
dataframe = pd.read_csv(url)

# Выбрать первую строку
dataframe.iloc[0]


# In[22]:


# Выбрать три строки
dataframe.iloc[1:4]


# In[23]:


# Выбрать три строки
dataframe.iloc[:4]


# In[24]:


# Задать индекс
dataframe = dataframe.set_index(dataframe['Name'])

# Показать строку
dataframe.loc['Allen, Miss Elisabeth Walton']


# > <b>3.4 Выбор строк на основе условных конструкций

# In[25]:


# Загрузить библиотеку 
import pandas as pd

# Создать URL-адрес 
url = 'https://tinyurl.com/titanic-csv' 

# Загрузить данные 
dataframe = pd.read_csv(url)

# Показать верхние две строки, где столбец 'sex' равняется 'female'
dataframe[dataframe['Sex'] == 'female'].head(2)


# In[26]:


# Отфильтровать строки
dataframe[(dataframe['Sex'] == 'female') & (dataframe['Age'] >= 65)]


# > <b>3.5 Замена значений

# In[27]:


# Загрузить библиотеку 
import pandas as pd

# Создать URL-адрес 
url = 'https://tinyurl.com/titanic-csv' 

# Загрузить данные 
dataframe = pd.read_csv(url)

# Заменить значения, показать две строки
dataframe['Sex'].replace("female", "Woman").head(2)


# In[28]:


# Заменить значения, показать две строки
dataframe.replace(1, "One").head(2)


# In[29]:


# Заменить значения, показать две строки
dataframe.replace(r"1st", "First", regex=True).head(2)


# > <b>3.6 Переименование столбцов

# In[30]:


# Загрузить библиотеку 
import pandas as pd

# Создать URL-адрес 
url = 'https://tinyurl.com/titanic-csv' 

# Загрузить данные 
dataframe = pd.read_csv(url)

# Переименовать столбец, показать две строки
dataframe.rename(columns={'PClass': 'Passenger Class'}).head(2)


# In[31]:


# Переименовать столбцы, показать две строки
dataframe.rename(columns={'PClass': 'Passenger Class', 'Sex': 'Gender'}).head(2)


# In[32]:


# Загрузить библиотеку
import collections

# Создать словарь
column_names = collections.defaultdict(str)

# Создать ключи
for name in dataframe.columns:
    column_names[name]

# Показать словарь
column_names


# > <b>3.7 Нахождение минимума, максимума, суммы, среднего арифметического и количества

# In[33]:


# Загрузить библиотеку 
import pandas as pd

# Создать URL-адрес 
url = 'https://tinyurl.com/titanic-csv' 

# Загрузить данные 
dataframe = pd.read_csv(url)

# Вычислить статистические показатели
print('Максимум:', dataframe['Age'].max())
print('Минимум:', dataframe['Age'].min())
print('Среднее:', dataframe['Age'].mean())
print('Сумма:', dataframe['Age'].sum())
print('Количество:', dataframe['Age'].count())


# In[34]:


# Показать количества значений
dataframe.count()


# > <b>3.8 Нахождение уникальных значений

# In[35]:


# Загрузить библиотеку
import pandas as pd

# Создать URL-адрес
url = 'https://tinyurl.com/titanic-csv' 

# Загрузить данные
dataframe = pd.read_csv(url)

# Выбрать уникальные значения
dataframe['Sex'].unique()


# In[36]:


# Показать количества появлений
dataframe['Sex'].value_counts()


# In[37]:


# Показать количества появлений
dataframe['PClass'].value_counts()


# In[38]:


# Показать количество уникальных значений
dataframe['PClass'].nunique()


# > <b>3.9 Отбор пропущенных значений

# In[39]:


# Загрузить библиотеку 
import pandas as pd

# Создать URL-адрес 
url = 'https://tinyurl.com/titanic-csv' 

# Загрузить данные 
dataframe = pd.read_csv(url)

# Выбрать пропущенные значения, показать две строки
dataframe[dataframe['Age'].isnull()].head(2)


# In[40]:


# Попытаться заменить значения с NaN
dataframe['Sex'] = dataframe['Sex'].replace('male', NaN)


# In[41]:


import numpy as np

# Заменить значения с NaN
dataframe['Sex'] = dataframe['Sex'].replace('male', np.nan)


# In[42]:


# Загрузить данные, задать пропущенные значения
dataframe = pd.read_csv(url, na_values=[np.nan, 'NONE', -999])


# > <b>3.10 Удаление столбца

# In[43]:


# Загрузить библиотеку
import pandas as pd

# Создать URL-адрес
url = 'https://tinyurl.com/titanic-csv'

# Загрузить данные
dataframe = pd.read_csv(url)

# Удалить столбец
dataframe.drop('Age', axis=1).head(2)


# In[45]:


# Исключить столбцы
dataframe.drop(['Age', 'Sex'], axis=1).head(2)


# In[46]:


# Исключить столбец
dataframe.drop(dataframe.columns[1], axis=1).head(2)


# In[48]:


# Создать новый фрейм данных
dataframe_name_dropped = dataframe.drop(dataframe.columns[0], axis=1)

dataframe_name_dropped.head()


# > <b>3.11 Удаление строки

# In[49]:


# Загрузить библиотеку 
import pandas as pd

# Создать URL-адрес 
url = 'https://tinyurl.com/titanic-csv'

# Загрузить данные 
dataframe = pd.read_csv(url)

# Удалить строки, показать первые две строки вывода
dataframe[dataframe['Sex'] != 'male'].head(2)


# In[50]:


# Удалить строку, показать первые две строки вывода
dataframe[dataframe['Name'] != 'Allison, Miss Helen Loraine'].head(2)


# In[51]:


# Удалить строку, показать первые две строки вывода
dataframe[dataframe.index != 0].head(2)


# > <b>3.12 Удаление повторяющихся строк

# In[52]:


# Загрузить библиотеку
import pandas as pd

# Создать URL-адрес
url = 'https://tinyurl.com/titanic-csv'

# Загрузить данные
dataframe = pd.read_csv(url)

# Удалить дубликаты, показать первые две строки вывода
dataframe.drop_duplicates().head(2)


# In[53]:


# Показать количество строк
print("Количество строк в исходном фрейме данных:", len(dataframe))
print("Количество строк после дедубликации:", len(dataframe.drop_duplicates()))


# In[54]:


# Удалить дубликаты
dataframe.drop_duplicates(subset=['Sex'])


# In[55]:


# Удалить дубликаты
dataframe.drop_duplicates(subset=['Sex'], keep='last')


# > <b>3.13 Группирование строк по значениям

# In[56]:


# Загрузить библиотеку
import pandas as pd

# Создать URL-адрес
url = 'https://tinyurl.com/titanic-csv'

# Загрузить данные
dataframe = pd.read_csv(url)

# Сгруппировать строки по значениям столбца 'Sex', вычислить среднее
# каждой группы
dataframe.groupby('Sex').mean()


# In[57]:


# Сгруппировать строки
dataframe.groupby('Sex')


# In[58]:


# Сгруппировать строки, подсчитать строки
dataframe.groupby('Survived')['Name'].count()


# In[59]:


# Сгруппировать строки, вычислить среднее
dataframe.groupby(['Sex','Survived'])['Age'].mean()


# > <b>3.14 Группирование строк по времени

# In[60]:


# Загрузить библиотеки 
import pandas as pd
import numpy as np

# Создать диапазон дат
time_index = pd.date_range('06/06/2017', periods=100000, freq='30S')

# Создать фрейм данных
dataframe = pd.DataFrame(index=time_index)

# Создать столбец случайных значений
dataframe['Sale_Amount'] = np.random.randint(1, 10, 100000)

# Сгруппировать строки по неделе, вычислить сумму за неделю
dataframe.resample('W').sum()


# In[61]:


# Показать три строки
dataframe.head(3)


# In[62]:


# Сгруппировать по двум неделям, вычислить среднее
dataframe.resample('2W').mean()


# In[63]:


# Сгруппировать по месяцу, побсчитать строки
dataframe.resample('M').count()


# In[64]:


# Сгруппировать по месяцу, побсчитать строки
dataframe.resample('M', label='left').count()


# > <b>3.15 Обход столбца в цикле

# In[65]:


# Загрузить библиотеку
import pandas as pd

# Создать URL-адрес
url = 'https://tinyurl.com/titanic-csv'

# Загрузить данные
dataframe = pd.read_csv(url)

# Напечатать первые два имени в верхнем регистре
for name in dataframe['Name'][0:2]:
    print(name.upper())


# In[66]:


# Напечатать первые два имени в верхнем регистре
[name.upper() for name in dataframe['Name'][0:2]]


# > <b>3.16 Применение функции ко всем элементам в столбце

# In[67]:


# Загрузить библиотеку
import pandas as pd

# оздать URL-адрес
url = 'https://tinyurl.com/titanic-csv'

# Загрузить данные
dataframe = pd.read_csv(url)

# Создать функцию
def uppercase(x):
    return x.upper()

# Применить функцию, показать две строки
dataframe['Name'].apply(uppercase)[0:2]


# > <b>3.17 Применение функции к группам

# In[68]:


# Загрузить библиотеку
import pandas as pd

# Создать URL-адрес
url = 'https://tinyurl.com/titanic-csv'

# Загрузить данные
dataframe = pd.read_csv(url)

# Сгруппировать строки, применить функцию к группам
dataframe.groupby('Sex').apply(lambda x: x.count())


# > <b>3.18 Конкатенация фреймов данных

# In[69]:


# Load library
import pandas as pd

# Создать фрейм данных
data_a = {'id': ['1', '2', '3'],
          'first': ['Alex', 'Amy', 'Allen'],
          'last': ['Anderson', 'Ackerman', 'Ali']}
dataframe_a = pd.DataFrame(data_a, columns = ['id', 'first', 'last'])

# Создать фрейм данных
data_b = {'id': ['4', '5', '6'],
          'first': ['Billy', 'Brian', 'Bran'],
          'last': ['Bonder', 'Black', 'Balwner']}
dataframe_b = pd.DataFrame(data_b, columns = ['id', 'first', 'last'])

# Конкатенировать фреймы данных построчно
pd.concat([dataframe_a, dataframe_b], axis=0)


# In[70]:


# Конкатенировать фреймы данных по столбцам
pd.concat([dataframe_a, dataframe_b], axis=1)


# In[71]:


# Создать строку
row = pd.Series([10, 'Chris', 'Chillon'], index=['id', 'first', 'last'])

# Добавить строку в конец
dataframe_a.append(row, ignore_index=True)


# > <b>3.19 Слияние фреймов данных

# In[72]:


# Загрузить библиотеку
import pandas as pd

# Создать фрейм данных
employee_data = {'employee_id': ['1', '2', '3', '4'],
                 'name': ['Amy Jones', 'Allen Keys', 'Alice Bees',
                          'Tim Horton']}
dataframe_employees = pd.DataFrame(employee_data, columns = ['employee_id',
                                                             'name'])

# Создать фрейм данных
sales_data = {'employee_id': ['3', '4', '5', '6'],
              'total_sales': [23456, 2512, 2345, 1455]}
dataframe_sales = pd.DataFrame(sales_data, columns = ['employee_id',
                                                      'total_sales'])

# Выполнить слияние фреймов данных
pd.merge(dataframe_employees, dataframe_sales, on='employee_id')


# In[73]:


# Выполнить слияние фреймов данных
pd.merge(dataframe_employees, dataframe_sales, on='employee_id', how='outer')


# In[74]:


# Выполнить слияние фреймов данных
pd.merge(dataframe_employees, dataframe_sales, on='employee_id', how='left')


# In[75]:


# Выполнить слияние фреймов данных
pd.merge(dataframe_employees,
         dataframe_sales,
         left_on='employee_id',
         right_on='employee_id')

