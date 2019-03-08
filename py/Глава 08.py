
# coding: utf-8

# # Глава 8. 
# ## Работа с изображениями
# > <b>8.1 Загрузка изображений

# *Определение стиля изображения*

# In[1]:


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


# In[2]:


# Загрузить библиотеки
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Загрузить изображение в оттенках серого
image = cv2.imread("images/plane.jpg", cv2.IMREAD_GRAYSCALE)

# Показать изображение
plt.imshow(image, cmap="gray"), plt.axis("off")
plt.show()


# In[4]:


# Показать тип данных
type(image)


# In[5]:


# Показать данные изображения
image


# In[6]:


# Показать размерности
image.shape


# In[7]:


# Показать первый пиксел
image[0,0]


# In[8]:


# Загрузить изображение в цвете
image_bgr = cv2.imread("images/plane.jpg", cv2.IMREAD_COLOR)

# Показать пиксел
image_bgr[0,0]


# In[9]:


# Конвертировать в RGB
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Показать изображение
plt.imshow(image_rgb), plt.axis("off")
plt.show()


# > <b>8.2 Сохранение изображений

# In[10]:


# Загрузить библиотеки
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Загрузить изображение в оттенках серого
image = cv2.imread("images/plane.jpg", cv2.IMREAD_GRAYSCALE)

# Сохранить изображение
cv2.imwrite("images/plane_new.jpg", image)


# > <b>8.3 Изменение размера изображений

# In[11]:


# Загрузить библиотеки
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Загрузить изображение в оттенках серого
image = cv2.imread("images/plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)

# Изменить размер изображения до 50 пикселов на 50 пикселов
image_50x50 = cv2.resize(image, (50, 50))

# Взглянуть на изображение
plt.imshow(image_50x50, cmap="gray"), plt.axis("off")
plt.show()


# > <b>8.4 Обрезка изображений

# In[12]:


# Загрузить библиотеки
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Загрузить изображение в оттенках серого
image = cv2.imread("images/plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)

# Выбрать первую половину столбцов и все строки
image_cropped = image[:,:128]

# Показать изображение
plt.imshow(image_cropped, cmap="gray"), plt.axis("off")
plt.show()


# > <b>8.5 Размытие изображений

# In[13]:


# Загрузить библиотеки
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Загрузить изображение в оттенках серого
image = cv2.imread("images/plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)

# Размыть изображение
image_blurry = cv2.blur(image, (5,5))

# Показать изображение
plt.imshow(image_blurry, cmap="gray"), plt.axis("off")
plt.show()


# In[14]:


# Размыть изображение
image_very_blurry = cv2.blur(image, (100,100))

# Показать изображение
plt.imshow(image_very_blurry, cmap="gray"), plt.xticks([]), plt.yticks([])
plt.show()


# In[15]:


# Создать ядро
kernel = np.ones((5,5)) / 25.0

# Показать ядро
kernel


# In[16]:


# Применить ядро
image_kernel = cv2.filter2D(image, -1, kernel)

# Показать изображение
plt.imshow(image_kernel, cmap="gray"), plt.xticks([]), plt.yticks([])
plt.show()


# > <b>8.6 Увеличение резкости изображений

# In[17]:


# Загрузить библиотеки
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Загрузить изображение в оттенках серого
image = cv2.imread("images/plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)

# Создать ядро
kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])

# Увеличить резкость изображения
image_sharp = cv2.filter2D(image, -1, kernel)

# Показать изображение
plt.imshow(image_sharp, cmap="gray"), plt.axis("off")
plt.show()


# > <b>8.7 Повышение контрастности

# In[18]:


# Загрузить библиотеки
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Загрузить изображение в оттенках серого
image = cv2.imread("images/plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)

# Улучшить изображение
image_enhanced = cv2.equalizeHist(image)

# Показать изображение
plt.imshow(image_enhanced, cmap="gray"), plt.axis("off")
plt.show()


# In[19]:


# Загрузить изображение
image_bgr = cv2.imread("images/plane.jpg")

# Конвертировать в YUV
image_yuv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YUV)

# Применить выравнивание гистограммы
image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])

# Конвертировать в RGB
image_rgb = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)

# Показать изображение
plt.imshow(image_rgb), plt.axis("off")
plt.show()


# > <b>8.8 Выделение цвета

# In[20]:


# Загрузить библиотеки
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Загрузить изображение
image_bgr = cv2.imread('images/plane_256x256.jpg')

# Конвертировать BGR в HSV
image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

# Определить диапазон синих значений в HSV
lower_blue = np.array([50,100,50])
upper_blue = np.array([130,255,255])

# Создать маску
mask = cv2.inRange(image_hsv, lower_blue, upper_blue)

# Наложить маску на изображение
image_bgr_masked = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)

# Конвертировать BGR в RGB
image_rgb = cv2.cvtColor(image_bgr_masked, cv2.COLOR_BGR2RGB)

# Показать изображение
plt.imshow(image_rgb), plt.axis("off")
plt.show()


# In[21]:


# Показать изображение
plt.imshow(mask, cmap='gray'), plt.axis("off")
plt.show()


# > <b>8.9 Бинаризация изображений

# In[22]:


# Загрузить библиотеки
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Загрузить изображение в оттенках серого
image_grey = cv2.imread("images/plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)

# Применить адаптивную пороговую обработку
max_output_value = 255
neighborhood_size = 99
subtract_from_mean = 10
image_binarized = cv2.adaptiveThreshold(image_grey,
                                        max_output_value,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY,
                                        neighborhood_size,
                                        subtract_from_mean)

# Показать изображение
plt.imshow(image_binarized, cmap="gray"), plt.axis("off")
plt.show()


# In[23]:


# Применить cv2.ADAPTIVE_THRESH_MEAN_C
image_mean_threshold = cv2.adaptiveThreshold(image_grey,
                                             max_output_value,
                                             cv2.ADAPTIVE_THRESH_MEAN_C,
                                             cv2.THRESH_BINARY,
                                             neighborhood_size,
                                             subtract_from_mean)

# Показать изображение
plt.imshow(image_mean_threshold, cmap="gray"), plt.axis("off")
plt.show()


# > <b>8.10 Удаление фонов

# In[24]:


# Загрузить библиотеки
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Загрузить изображение и конвертировать в RGB
image_bgr = cv2.imread('images/plane_256x256.jpg')
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Значения прямоугольника: началная x, начальная y, ширина, высота
rectangle = (0, 56, 256, 150)

# Создать первоначальную маску
mask = np.zeros(image_rgb.shape[:2], np.uint8)

# Создать временные маасивы, используемые в grabCut
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# Выполнить алгоритм grabCut
cv2.grabCut(image_rgb, # Наше изображение
            mask,      # Маска
            rectangle, # Наш прямоугольник
            bgdModel,  # Временный массив для фона
            fgdModel,  # Временный массив для переднего плана
            5,         # Количество итераций
            cv2.GC_INIT_WITH_RECT) # Инициализировать, используя прямоугольник

# Создать маску, где фоны уверенно или потенциально равны 0, иначе 1
mask_2 = np.where((mask==2) | (mask==0), 0, 1).astype('uint8')

# Умножить изображение на новую маску, чтобы вычесть фон
image_rgb_nobg = image_rgb * mask_2[:, :, np.newaxis]

# Показать изображение
plt.imshow(image_rgb_nobg), plt.axis("off")
plt.show()


# In[25]:


# Показать маску
plt.imshow(mask, cmap='gray'), plt.axis("off")
plt.show()


# In[26]:


# Показать маску
plt.imshow(mask_2, cmap='gray'), plt.axis("off")
plt.show()


# > <b>8.11 Обнаружение краев

# In[27]:


# Загрузить библиотеки
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Загрузить изображение в оттенках серого
image_gray = cv2.imread("images/plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)

# Вычислить медиану интенсивности
median_intensity = np.median(image_gray)

# Установить пороговые значения на одно стандартное отклонение 
# выше и ниже медианы интенсивности
lower_threshold = int(max(0, (1.0 - 0.33) * median_intensity))
upper_threshold = int(min(255, (1.0 + 0.33) * median_intensity))

# Применить детектор границ Кэнни
image_canny = cv2.Canny(image_gray, lower_threshold, upper_threshold)

# Показать изображение
plt.imshow(image_canny, cmap="gray"), plt.axis("off")
plt.show()


# > <b>8.12 Обнаружение углов

# In[28]:


# Загрузить библиотеки
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Загрузить изображение в оттенках серого
image_bgr = cv2.imread("images/plane_256x256.jpg")
image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
image_gray = np.float32(image_gray)

# Задать параметры детектора углов
block_size = 2
aperture = 29
free_parameter = 0.04

# Обнаружить углы
detector_responses = cv2.cornerHarris(image_gray,
                                      block_size,
                                      aperture,
                                      free_parameter)

# Крупные угловые маркеры
detector_responses = cv2.dilate(detector_responses, None)

# Оставить только те отклики детектора, которые больше порога, 
# пометить белым цветом
threshold = 0.02
image_bgr[detector_responses >
          threshold *
          detector_responses.max()] = [255,255,255]

# Конвертировать в оттенки серого
image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

# Показать изображение
plt.imshow(image_gray, cmap="gray"), plt.axis("off")
plt.show()


# In[29]:


# Показать потенциальные углы
plt.imshow(detector_responses, cmap='gray'), plt.axis("off")
plt.show()


# In[30]:


# Загрузить изображения
image_bgr = cv2.imread('images/plane_256x256.jpg')
image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

# Количество углов для обнаружения
corners_to_detect = 10
minimum_quality_score = 0.05
minimum_distance = 25

# Обнаружить углы
corners = cv2.goodFeaturesToTrack(image_gray,
                                  corners_to_detect,
                                  minimum_quality_score,
                                  minimum_distance)
corners = np.float32(corners)

# Нарисовать белый круг в каждом углу 
for corner in corners:
    x, y = corner[0]
    cv2.circle(image_bgr, (x,y), 10, (255,255,255), -1)

# Конвертировать в оттенки серого
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

# Показать изображение
plt.imshow(image_rgb, cmap='gray'), plt.axis("off")
plt.show()


# > <b>8.13 Создание признаков для машинного самообучения

# In[31]:


# Загрузить библиотеки
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Загрузить изображение в оттенках серого
image = cv2.imread("images/plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)

# Изменить размер изображения до 10 пикселов на 10 пикселов
image_10x10 = cv2.resize(image, (10, 10))

# Конвертировать данные изображения в одномерный вектор
image_10x10.flatten()


# In[32]:


plt.imshow(image_10x10, cmap="gray"), plt.axis("off")
plt.show()


# In[33]:


image_10x10.shape


# In[34]:


image_10x10.flatten().shape


# In[ ]:


# Загрузить изображение в цвете
image_color = cv2.imread("images/plane_256x256.jpg", cv2.IMREAD_COLOR)

# Изменить размер изображения до 10 пикселов на 10 пикселов
image_color_10x10 = cv2.resize(image_color, (10, 10))

# Конвертировать данные изображения в одномерный вектор, 
# показать размерности
image_color_10x10.flatten().shape


# In[35]:


# Загрузить изображение в оттенках серого
image_256x256_gray = cv2.imread("images/plane_256x256.jpg", 
                                cv2.IMREAD_GRAYSCALE)

# Конвертировать данные изображения в одномерный вектор, 
# показать размерности
image_256x256_gray.flatten().shape


# In[36]:


# Загрузить изображение в цвете
image_256x256_color = cv2.imread("images/plane_256x256.jpg", 
                                 cv2.IMREAD_COLOR)

# Конвертировать данные изображения в одномерный вектор, 
# показать размерности
image_256x256_color.flatten().shape


# > <b>8.14 Кодирование среднего цвета в качестве признака

# In[37]:


# Загрузить библиотеки
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Загрузить изображение как BGR
image_bgr = cv2.imread("images/plane_256x256.jpg", cv2.IMREAD_COLOR)

# Вычислить среднее значение каждого канала
channels = cv2.mean(image_bgr)

# Поменять местами синее и красное значения (переведя в RGB, не BGR)
observation = np.array([(channels[2], channels[1], channels[0])])

# Показать значения среднего канала
observation


# In[38]:


# Показать изображение
plt.imshow(observation), plt.axis("off")
plt.show()


# > <b>8.15 Кодирование гистограмм цветовых каналов в качестве признаков

# In[10]:


# Загрузить библиотеки
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Загрузить изображение
image_bgr = cv2.imread("images/plane_256x256.jpg", cv2.IMREAD_COLOR)

# Конвертировать в RGB
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Создать список для значений признаков
features = []

# Вычислить гистограмму для каждого цветового канала
colors = ("r","g","b")

# Для каждого цветового канала: 
# вычислить гистограмму и добавить в список значений признаков
for i, channel in enumerate(colors):
    histogram = cv2.calcHist([image_rgb], # Изображение
                             [i],         # Индекс канала
                             None,        # Маска отсутствует
                             [256],       # Размер гистограммы
                             [0,256])     # Диапазон
    features.extend(histogram)

# Создать вектор для значений признаков наблюдения
observation = np.array(features).flatten()

# Показать значение наблюдения для первых пяти признаков
observation[0:5]


# In[11]:


# Показать значения канала RGB
image_rgb[0,0]


# In[12]:


# Импортировать pandas
import pandas as pd

# Создать немного данных
data = pd.Series([1, 1, 2, 2, 3, 3, 3, 4, 5])

# Показать гистограмму
data.hist(grid=False)
plt.tight_layout()
plt.savefig('pics/8_15.png', dpi=600) 
plt.show()


# In[13]:


# Вычислить гистограмму для каждого цветового канала
colors = ("r","g","b")

# Для каждого канала: Вычислить гистограмму, построить график
for i, channel in enumerate(colors):
    histogram = cv2.calcHist([image_rgb], # Изображение
                             [i],         # Индекс канала
                             None,        # Маска отсутствует
                             [256],       # Размер гистограммы
                             [0,256])     # Диапазон
    plt.plot(histogram, color = channel)
    plt.xlim([0,256])

# Показать график
plt.tight_layout()
plt.savefig('pics/8_15_2.png', dpi=600) 
plt.show()

