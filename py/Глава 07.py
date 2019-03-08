
# coding: utf-8

# # Глава 7. 
# ## Работа с датой и временем
# > <b>7.1 Конвертирование строковых значений в даты

# In[5]:


# Загрузить библиотеки
import numpy as np
import pandas as pd

# Создать строки
date_strings = np.array(['03-04-2005 11:35 PM',
                         '23-05-2010 12:01 AM',
                         '04-09-2009 09:09 PM'])

# Конвертировать в метки datetime
[pd.to_datetime(date, format='%d-%m-%Y %I:%M %p') for date in date_strings]


# In[4]:


# Конвертировать в метки datetime
[pd.to_datetime(date, format="%d-%m-%Y %I:%M %p", errors="coerce")
for date in date_strings]


# > <b>7.2 Обработка часовых поясов

# In[6]:


# Загрузить библиотеку
import pandas as pd

# Создать метку datetime
pd.Timestamp('2017-05-01 06:00:00', tz='Europe/London')


# In[7]:


# Создать метку datetime
date = pd.Timestamp('2017-05-01 06:00:00')

# Задать часовой пояс
date_in_london = date.tz_localize('Europe/London')

# Показать метку datetime
date_in_london


# In[8]:


# Изменить часовой пояс
date_in_london.tz_convert('Africa/Abidjan')


# In[9]:


# Создать три даты
dates = pd.Series(pd.date_range('2/2/2002', periods=3, freq='M'))

# Задать часовой пояс
dates.dt.tz_localize('Africa/Abidjan')


# In[10]:


# Загрузить библиотеку
from pytz import all_timezones

# Показать два часовых пояса
all_timezones[0:2]


# > <b>7.3 Выбор дат и времени

# In[16]:


# Загрузить библиотеку
import pandas as pd

# Создать фрейм данных
dataframe = pd.DataFrame()

# Создать метки datetime
dataframe['дата'] = pd.date_range('1/1/2001', periods=100000, freq='H')

# Выбрать наблюдения между двумя метками datetime
dataframe[(dataframe['дата'] > '2002-1-1 01:00:00') &
(dataframe['дата'] <= '2002-1-1 04:00:00')]


# In[17]:


# Задать индекс
dataframe = dataframe.set_index(dataframe['дата'])

# Выбрать наблюдения между двумя метками datetime
dataframe.loc['2002-1-1 01:00:00':'2002-1-1 04:00:00']


# > <b>7.4 Разбиение данных даты на несколько признаков

# In[15]:


# Загрузить библиотеку
import pandas as pd

# Создать фрейм данных
dataframe = pd.DataFrame()

# Создать пять дат
dataframe['дата'] = pd.date_range('1/1/2001', periods=150, freq='W')

# Создать признаки для года, месяца, дня, часа и минуты
dataframe['год']    = dataframe['дата'].dt.year
dataframe['месяц']  = dataframe['дата'].dt.month
dataframe['день']   = dataframe['дата'].dt.day
dataframe['час']    = dataframe['дата'].dt.hour
dataframe['минута'] = dataframe['дата'].dt.minute

# Показать три строки фрейма
dataframe.head(3)


# > <b>7.5 Вычисление разницы между датами

# In[18]:


# Загрузить библиотеку
import pandas as pd

# Создать фрейм данных
dataframe = pd.DataFrame()

# Создать два признака datetime
dataframe['Прибыло'] = [pd.Timestamp('01-01-2017'), pd.Timestamp('01-04-2017')]
dataframe['Осталось'] = [pd.Timestamp('01-01-2017'), pd.Timestamp('01-06-2017')]

# Вычислить продолжительность между признаками
dataframe['Осталось'] - dataframe['Прибыло']


# In[19]:


# Вычислить продолжительность между признаками
pd.Series(delta.days 
          for delta in (dataframe['Осталось'] - dataframe['Прибыло']))


# > <b>7.6 Кодирование дней недели

# In[20]:


# Загрузить библиотеку
import pandas as pd

# Создать даты
dates = pd.Series(pd.date_range("2/2/2002", periods=3, freq="M"))

# Показать дни недели
dates.dt.weekday_name


# In[21]:


# Показать дни недели
dates.dt.weekday


# > <b>7.7 Создание запаздывающего признака

# In[22]:


# Загрузить библиотеку
import pandas as pd

# Создать фрейм данных
dataframe = pd.DataFrame()

# Создать дату
dataframe["даты"] = pd.date_range("1/1/2001", periods=5, freq="D")
dataframe["цена_акций"] = [1.1,2.2,3.3,4.4,5.5]

# Значения с запаздыванием на одну строку
dataframe["цена_акций_в_предыдущий_день"] = dataframe["цена_акций"].shift(1)

# Показать фрейм данных
dataframe


# > <b>7.8 Использование скользящих временных окон

# In[23]:


# Загрузить библиотеку
import pandas as pd

# Создать метки datetime
time_index = pd.date_range("01/01/2010", periods=5, freq="M")

# Создать фрейм данных, задать индекс
dataframe = pd.DataFrame(index=time_index)

# Создать признак
dataframe["цена_акций"] = [1,2,3,4,5]

# Вычислить скользящее среднее 
dataframe.rolling(window=2).mean()


# > <b>7.9 Обработка пропущенных дат во временном ряду

# In[27]:


# Загрузить библиотеки
import pandas as pd
import numpy as np

# Создать дату
time_index = pd.date_range("01/01/2010", periods=5, freq="M")

# Создать фрейм данных, задать индекс
dataframe = pd.DataFrame(index=time_index)

# Создать признак с промежутком пропущенных значений
dataframe["продажи"] = [1.0,2.0,np.nan,np.nan,5.0]

# Интерполировать пропущенные значения
dataframe.interpolate()


# In[28]:


# Создать признак с промежутком пропущенных значений
dataframe["продажи"] = [1.0,2.0,np.nan,np.nan,5.0]

# Прямое заполнение
dataframe.ffill()


# In[29]:


# Создать признак с промежутком пропущенных значений
dataframe["продажи"] = [1.0,2.0,np.nan,np.nan,5.0]

# Обратное заполнение
dataframe.bfill()


# In[30]:


# Интерполировать пропущенные значения
dataframe.interpolate(method="quadratic")


# In[31]:


# Интерполировать пропущенные значения
dataframe.interpolate(limit=1, limit_direction="forward")

