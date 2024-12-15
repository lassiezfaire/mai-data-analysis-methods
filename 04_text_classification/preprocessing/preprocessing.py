import pickle as pk
import os
import re

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

pd.set_option('display.max_columns', None)

def num_classes(filepath: str) -> int:
    """ Функция, определяющая количество классов

    :param filepath: путь к файлу с названиями классов
    :return: количество классов
    """
    count = 0
    with open(filepath, 'r') as f:
      for _ in f:
        count += 1
      return count

def clean_text(text):
    # Приведение к нижнему регистру
    text = text.lower()
    # Удаление специальных символов и цифр
    text = re.sub(r'[^a-z\s]', '', text)
    return text

class DataFrame:
    def __init__(self, csv_path: str, tokenizer: Tokenizer, max_length, num_classes: int):
        """Класс, осуществляющий первичную предобработку данных для дальнейшего обучения

        :param csv_path: путь к csv-файлу с данными
        :param tokenizer: токенизатор
        :param max_length: максимальная длина текста, предназначенного для токенизации
        :param num_classes: количество классов в датасете
        """

        self.df = pd.read_csv(csv_path, names=['Class', 'Title', 'Text'])

        # очищаем текст
        self.df['Text'] = self.df['Text'].apply(clean_text)

        # кодировка классов
        label_encoder = LabelEncoder()
        self.df['Class'] = label_encoder.fit_transform(self.df['Class'])

        # Преобразование меток в one-hot encoding
        # self.df['Class'] = to_categorical(self.df['Class'], num_classes=num_classes)
        self.df_labels = to_categorical(self.df['Class'], num_classes=num_classes)

        # Перемешиваем
        self.df = self.df.sample(frac=1, random_state=43).reset_index(drop=True)

        # Токенизируем
        self.tokenizer = tokenizer
        self.tokenizer.fit_on_texts(self.df['Text'])
        self.sequences = tokenizer.texts_to_sequences(self.df['Text'])

        with open(os.path.join('data', 'tokenizer_m1.pickle'), 'wb') as handle:
            pk.dump(self.tokenizer, handle, protocol=pk.HIGHEST_PROTOCOL)

        word_index = self.tokenizer.word_index
        print('Found %s unique tokens. ' % len(word_index))

        # паддинг
        self.df_padded = pad_sequences(self.sequences, maxlen=max_length, padding='post')
        print('Data Shape: {}'.format(self.df_padded.shape))
