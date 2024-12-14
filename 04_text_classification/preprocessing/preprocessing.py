import pandas as pd
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

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

class DataFrame:
    def __init__(self, csv_path: str, tokenizer: Tokenizer, max_length, num_classes: int):
        """Класс, осуществляющий первичную предобработку данных для дальнейшего обучения

        :param csv_path: путь к csv-файлу с данными
        :param tokenizer: токенизатор
        :param max_length: максимальная длина текста, предназначенного для токенизации
        :param num_classes: количество классов в датасете
        """

        self.df = pd.read_csv(csv_path, names=['Class', 'Title', 'Text'])

        label_encoder = LabelEncoder()
        self.df['Class'] = label_encoder.fit_transform(self.df['Class'])

        self.df = self.df.sample(frac=1, random_state=43).reset_index(drop=True)
        self.df_labels = to_categorical(self.df['Class'], num_classes)

        self.tokenizer = tokenizer.fit_on_texts(self.df['Text'])

        self.df_sequences = tokenizer.texts_to_sequences(self.df['Text'])

        self.df_padded = pad_sequences(self.df_sequences, maxlen=max_length, padding='post')
