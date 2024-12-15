import os
import re

import pandas as pd

import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')

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

def get_wordnet_pos (tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def lemmatize(word_list):
    wl = WordNetLemmatizer()
    word_pos_tags = pos_tag(word_list)
    lemmatized_list = []
    for tag in word_pos_tags:
        lemmatize_word = wl.lemmatize(tag[0],get_wordnet_pos(tag[1]))
        lemmatized_list.append(lemmatize_word)
    return " ".join(lemmatized_list)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)

    word_tokens = word_tokenize(text)
    stop_words = stopwords.words('english')
    text_tokens = [word for word in word_tokens if (len(word) > 2) and (word not in stop_words)]

    text = lemmatize(text_tokens)
    return text

def preprocessing(csv_name: str, clean_csv_name: str, path_to_data: str = 'data') -> tuple[pd.Series, pd.Series]:
    df = pd.read_csv(os.path.join(path_to_data, csv_name), names=['Class', 'Title', 'Text'])

    csv_path = os.path.join(path_to_data, clean_csv_name)

    if os.path.isfile(csv_path):
        df = pd.read_csv(csv_path, names=['Class', 'Title', 'Text'])
    else:
        df['Text'] = df['Text'].apply(clean_text)
        df.to_csv(csv_path, header=False, index=False)

    print(df.head(5))

    X = df['Text']
    y = df['Class']

    return X, y

X_train, y_train = preprocessing(csv_name='train.csv', clean_csv_name='clean_train.csv', path_to_data='..\\data')
X_test, y_test = preprocessing(csv_name='test.csv', clean_csv_name='clean_test.csv', path_to_data='..\\data')

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)
