
# coding: utf-8

# # Глава 6. 
# ## Работа с текстом
# > <b>6.1 Очистка текста

# In[1]:


# Создать текст
text_data = ["   Interrobang. By Aishwarya Henriette     ",
             "Parking And Going. By Karl Gautier",
             "    Today Is The night. By Jarek Prakash   "]

# Удалить пробелы
strip_whitespace = [string.strip() for string in text_data]

# Показать текст
strip_whitespace


# In[2]:


# Удалить точки
remove_periods = [string.replace(".", "") for string in strip_whitespace]

# Показать текст
remove_periods


# In[3]:


# Создать функцию
def capitalizer(string: str) -> str:
    return string.upper()

# Применить функцию
[capitalizer(string) for string in remove_periods]


# In[4]:


# Импортировать библиотеку
import re

# Создать функцию
def replace_letters_with_X(string: str) -> str:
    return re.sub(r"[a-zA-Z]", "X", string)

# Применить функцию
[replace_letters_with_X(string) for string in remove_periods]


# > <b>6.2 Разбор и очистка разметки HTML

# In[21]:


# Загрузить библиотеку
from bs4 import BeautifulSoup
import lxml

'''
https://www.crummy.com/software/BeautifulSoup/bs4/doc/
BeautifulSoup(markup, parser)
где parser = "html.parser", "lxml", "lxml-xml", "xml", "html5lib"
'''

# Создать немного кода с разметкой HTML
html = """
       <div class='full_name'><span style='font-weight:bold'>
       Masego</span> Azra</div>"
       """

# Выполнить разбор html
soup = BeautifulSoup(html, "html.parser")  # "lxml"

# Найти элемент div с классом "full_name", показать текст
soup.find("div", { "class" : "full_name" }).text


# > <b>6.3 Удаление знаков препинания

# In[22]:


# Загрузить библиотеки
import unicodedata
import sys

# Создать текст
text_data = ['Hi!!!! I. Love. This. Song....',
             '10000% Agree!!!! #LoveIT',
             'Right?!?!']

# Создать словарь знаков препинания
punctuation = dict.fromkeys(i for i in range(sys.maxunicode)
                            if unicodedata.category(chr(i)).startswith('P'))

# Удалить любые знаки препинания во всех строковых значениях
[string.translate(punctuation) for string in text_data]


# > <b>6.4 Лексемизация текста

# In[25]:


# Загрузить библиотеку
from nltk.tokenize import word_tokenize

# Создать текст
string = "Сегодняшняя наука - это технология завтрашнего дня"

# Лексемизировать на слова
word_tokenize(string)


# In[27]:


# Загрузить библиотеку
from nltk.tokenize import sent_tokenize

# Создать текст
string = """Сегодняшняя наука - это технология завтрашнего дня. 
            Затра начинается сегодня."""

# Лексемизировать на предложения
sent_tokenize(string)


# > <b>6.5 Удаление стоп-слов

# In[28]:


# Загрузить библиотеку
from nltk.corpus import stopwords

# Перед этим вам следует скачать набор стоп-слов
# import nltk
# nltk.download('stopwords')

# Создать словарные лексемы
tokenized_words = ['i',
                   'am',
                   'going',
                   'to',
                   'go',
                   'to',
                   'the',
                   'store',
                   'and',
                   'park']

# Загрузить стоп-слова
stop_words = stopwords.words('english')

# Удалить стоп-слова
[word for word in tokenized_words if word not in stop_words]


# In[29]:


# Удалить стоп-слова
stop_words[:5]


# In[45]:


stop_list = stopwords.words('russian')

# Создать словарные лексемы
tokenized_words = ['я',
                   'бы',
                   'пошел',
                   'в',
                   'пиццерию',
                   'покушать',
                   'пиццы',
                   'и',
                   'потом',
                   'в',
                   'парк']

# Загрузить стоп-слова
stop_words = stopwords.words('russian')

# Удалить стоп-слова
[word for word in tokenized_words if word not in stop_words]


# > <b>6.6 Выделение основ слов

# In[30]:


# Загрузить библиотеку
from nltk.stem.porter import PorterStemmer

# Создать лексемы слов
tokenized_words = ['i', 'am', 'humbled', 'by', 'this', 'traditional', 'meeting']

# Создать стеммер
porter = PorterStemmer()

# Применить стеммер
[porter.stem(word) for word in tokenized_words]


# In[36]:


# Загрузить библиотеку
from nltk.stem.snowball import SnowballStemmer

# Создать лексемы слов
tokenized_words = ['рыбаки', 'рыбаков', 'рыбаками']

# Создать стеммер
snowball = SnowballStemmer("russian")

# Применить стеммер
[snowball.stem(word) for word in tokenized_words]


# > <b>6.7 Лемматизация слов

# In[50]:


# Загрузить библиотеку
from nltk.stem import WordNetLemmatizer

# Создать лексемы слов
tokenized_words = ['go', 'went', 'gone', 'am', 'are', 'is', 'was', 'were']

# Создать лемматизатор
lemmatizer = WordNetLemmatizer()

# Применить лемматизатор
[lemmatizer.lemmatize(word, pos='v') for word in tokenized_words]  


# > <b>6.8 Разметка слов на части речи

# In[89]:


# Загрузить библиотеки
from nltk import pos_tag
from nltk import word_tokenize

# Создать текст
text_data = "Chris loved outdoor running"

# Использовать предварительно натренированный 
# разметчик частей речи
text_tagged = pos_tag(word_tokenize(text_data))

# Показать части речи
text_tagged


# In[90]:


# Отфильтровать слова
[word for word, tag in text_tagged if tag in ['NN','NNS','NNP','NNPS'] ]


# In[59]:


# Загрузить библиотеки
import nltk
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer

# Создать текст
tweets = ["I am eating a burrito for breakfast",
          "Political science is an amazing field",
          "San Francisco is an awesome city"]

# Создать список
tagged_tweets = []

# Пометить каждое слово и каждый твит
for tweet in tweets:
    tweet_tag = nltk.pos_tag(word_tokenize(tweet))
    tagged_tweets.append([tag for word, tag in tweet_tag])

# Применить кодирование с одним активным состоянием, чтобы
# конвертировать метки в признаки
one_hot_multi = MultiLabelBinarizer()
one_hot_multi.fit_transform(tagged_tweets)


# In[60]:


# Показать имена признаков
one_hot_multi.classes_


# In[85]:


# Загрузить библиотеки
import nltk
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer

# Создать текст
tweets = ["Утро было ясным и теплым, и мы решили пойти в парк"]

# Создать список
tagged_tweets = []

# Пометить каждое слово и каждый твит
for tweet in tweets:
    tweet_tag = nltk.pos_tag(word_tokenize(tweet))
    tagged_tweets.append([tag for word, tag in tweet_tag])

# Применить кодирование с одним активным состоянием, чтобы
# конвертировать метки в признаки
one_hot_multi = MultiLabelBinarizer()
one_hot_multi.fit_transform(tagged_tweets)


# In[86]:


# Показать имена признаков
one_hot_multi.classes_


# In[43]:


# Загрузить библиотеку
from nltk.corpus import brown
from nltk.tag import UnigramTagger
from nltk.tag import BigramTagger
from nltk.tag import TrigramTagger

# Получить немного текста из стандартного текстового корпуса 
# Brown Corpus, разбитого на предложения
sentences = brown.tagged_sents(categories='news')

# Выделить на 4000 предложений для тренировки и 623 для тестирования
train = sentences[:4000]
test = sentences[4000:]

# Создать разметчик с откатом 
unigram = UnigramTagger(train)
bigram = BigramTagger(train, backoff=unigram)
trigram = TrigramTagger(train, backoff=bigram)

# Показать точность
trigram.evaluate(test)


# > <b>6.9 Кодирование текста в качестве мешка слов

# In[95]:


# Загрузить библиотеки
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# Создать текст
text_data = np.array(['Бразилия – моя любовь. Бразилия!',
                      'Швеция - лучше',
                      'Германия бьет обоих'])

'''
В оригинале:
text_data = np.array(['I love Brazil. Brazil!',
                      'Sweden is best',
                      'Germany beats both'])
'''

# Создать матрицу признаков на основе мешка слов
count = CountVectorizer()
bag_of_words = count.fit_transform(text_data)

# Показать матрицу признаков
bag_of_words


# In[96]:


bag_of_words.toarray()


# In[97]:


# Показать имена признаков
count.get_feature_names()


# In[100]:


# Создать матрицу признаков с аргументами
count_2gram = CountVectorizer(ngram_range=(1,2),
                              stop_words="english",
                              vocabulary=['бразилия'])

bag = count_2gram.fit_transform(text_data)

# Взглянуть на матрицу признаков
bag.toarray()


# In[101]:


# Взглянуть на 1-граммы и 2-граммы
count_2gram.vocabulary_


# > <b>6.10 Взвешивание важности слов

# In[105]:


# Загрузить библиотеки
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Создать текст
text_data = np.array(['Бразилия – моя любовь. Бразилия!',
                      'Швеция - лучше',
                      'Германия бьет обоих'])

# Создать матрицу признаков на основе меры tf-idf
tfidf = TfidfVectorizer()
feature_matrix = tfidf.fit_transform(text_data)

# Показать матрицу признаков на основе меры tf-idf
feature_matrix


# In[106]:


# Показать матрицу признаков на основе меры tf-idf как плотную
feature_matrix.toarray()


# In[107]:


# Показать имена признаков
tfidf.vocabulary_

