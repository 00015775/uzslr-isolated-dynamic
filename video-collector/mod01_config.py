import os

# 1. GLOBAL SETTINGS
DATA_ROOT = "./Data_Numpy_Arrays_RSL_UzSL"        
VIDEO_DEVICE = 1                                  
FRAME_WIDTH, FRAME_HEIGHT = 1280, 720
FPS = 30
FRAMES_PER_REP = 32 # fixed length per repetition
COUNTDOWN_SECONDS = 2
# make sure that the number of frames is divisible by 2


# 2. INITIAL SIGN-WORD LIST (initially there was 100 words)
# INITIAL_SIGNS = [
#     'uy', 'maktab', "ko'cha", 'eshik', 'stol', 'stul', 'karavot', 'mashina', 'poezd',
#     'metro', 'kema', "ko'prik", "yomg'ir", 'qor', 'shamol', 'qish', 'bahor', 'kun',
#     'iltimos', 'assalomu_alaykum', 'kechirasiz', 'yaxshi', 'tez', 'birga', 'katta',
#     'kichik', 'yangi', 'mehribon', 'ovqat_tayyorlash', 'yozish', 'yordam_berish', "o'ynash", 'sayr_qilish',
#     'qidirish', "yo'qotish", 'jismoniy_tarbiya', 'turish', 'ketish', 'olib_kelish', 'ochish',
#     'yopish', "yorug'", "qorong'i", 'toza', 'televizor', "to'la", "bo'sh", 'oxiri',
#     'boshlanishi', 'kech', 'tez_orada', 'restoran', 'zavod', 'masjid', 'bozor', 'shokolad',
#     'mehmonxona', 'kitob', "qog'oz", 'parda', 'kiyim', 'oyoq_kiyim', 'paypoq', "qo'lqop",
#     'bosh_kiyim', 'bank_kartasi', 'non', 'likopcha', 'muzlatkich', 'internet', 'musiqa',
#     'javob', 'yer', "o't", 'tosh', 'it', 'mushuk', 'sigir', 'ot', "qo'y",
#     "cho'chqa", 'kartoshka', 'sabzi', 'karam', 'pomidor', 'bodring', 'sarimsoq', "qo'ziqorin",
#     "sariyog'", 'stakan', 'futbol', "o'qiyman", "o'simlik_yog'i", 'sovun', 'yostiq',
#     'quyon', 'tozalayman', 'topaman', 'kir', 'ryukzak'
# ]

# After data cleaning and prunning, 50 signs were left. CLEAN_SIGNS
# MAKE SURE TO KEEP THIS DICTIONARY UPDATED IN CASE IF SIGNS ARE ADDED OR REMOVED
DEFAULT_SIGNS = ['assalomu_alaykum', 'bahor', 'birga', "bo'sh", 'bosh_kiyim', 'boshlanishi', 'bozor', 'eshik', 
               'futbol', 'iltimos', 'internet', 'javob', 'jismoniy_tarbiya', 'karam', 'kartoshka', 
               'kichik', 'kitob', "ko'prik", 'likopcha', 'maktab', 'mehmonxona', 'mehribon', 'metro', 
               'musiqa', "o'simlik_yog'i", "o'ynash", 'ochish', 'ot', 'ovqat_tayyorlash', 
               'oxiri', 'poezd', 'pomidor', 'qidirish', 'qish', "qo'ziqorin", 'qor', "qorong'i", 'quyon', 
               'restoran', "sariyog'", 'shokolad', 'sovun', 'stakan', 'televizor', 'tosh', 'toza',
               'turish', "yomg'ir", 'yopish', 'yordam_berish']


# Top 100 most used Signs in Tashkent defined by the original signer01
top_100_words = ['дом', 'школа', 'улица', 'дверь', 'стол', 'стул', 'кровать', 'машина', 'поезд', 
                 'метро', 'корабль', 'мост', 'дождь', 'снег', 'ветер', 'зима', 'весна', 'день', 
                 'пожалуйста', 'здравствуйте', 'извините', 'нормально', 'быстро', 'вместе', 'большой', 
                 'маленький', 'новый', 'добрый', 'готовить', 'писать', 'помогать', 'играть', 'гулять', 
                 'искать', 'терять', 'физкультура', 'вставать', 'уходить', 'приносить', 'открывать', 
                 'закрывать', 'светлый', 'тёмный', 'чистый', 'телевизор', 'полный', 'пустой', 'конец', 
                 'начало', 'поздно', 'скоро', 'ресторан', 'завод', 'мечеть', 'базар', 'шоколад', 
                 'гостиница', 'книга', 'бумага', 'занавеска', 'одежда', 'обувь', 'носки', 'перчатки', 
                 'шапка', 'банковская_карта', 'хлеб', 'тарелка', 'холодильник', 'интернет', 'музыка', 
                 'ответ', 'земля', 'трава', 'камень', 'собака', 'кошка', 'корова', 'лошадь', 'овца', 
                 'свинья', 'картошка', 'морковь', 'капуста', 'помидор', 'огурец', 'чеснок', 'грибы', 
                 'сливочное_масло', 'стакан', 'футбол', 'читаю', 'подсолнечное_масло', 'мыло', 'подушка', 
                 'кролик', 'убираю', 'нахожу', 'грязный', 'рюкзак']


# 3. MEDIA-PIPE SETTINGS
MP_CONFIDENCE = 0.5
POSE_REMOVE_IDX = [0,1,2,3,4,5,6,7,8,9,10,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
POSE_KEEP_CONNECTIONS = frozenset([(11,12),(11,13),(12,14),(13,15),(14,16)])

