import matplotlib.pyplot as plt
import pandas as pd

from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, GRU, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

from preprocessing import preprocessing
from models.cnn import cnn_model
from models.lstm import lstm_model
from models.gru import gru_model

# Загрузка данных
train_df, test_df, classes = preprocessing(
    train_path='data\\train.csv',
    test_path='data\\test.csv',
    classes_path='data\\classes.txt'
)

# Проверка структуры данных
print(train_df.head())
print(test_df.head())
print(classes.head())

# Токенизация текста
tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(train_df['Text'])

train_sequences = tokenizer.texts_to_sequences(train_df['Text'])
test_sequences = tokenizer.texts_to_sequences(test_df['Text'])

# Паддинг последовательностей
max_length = 100
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='post')
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post')

# one-hot encoding
num_classes = len(classes)
train_labels = to_categorical(train_df['Class'], num_classes)
test_labels = to_categorical(test_df['Class'], num_classes)

cnn_model = cnn_model(num_classes, max_length)
lstm_model = lstm_model(num_classes, max_length)
gru_model = gru_model(num_classes, max_length)

epochs = 5
batch_size = 64

cnn_history = cnn_model.fit(train_padded, train_labels, epochs=epochs, batch_size=batch_size, validation_split=0.2)
lstm_history = lstm_model.fit(train_padded, train_labels, epochs=epochs, batch_size=batch_size, validation_split=0.2)
gru_history = gru_model.fit(train_padded, train_labels, epochs=epochs, batch_size=batch_size, validation_split=0.2)

# Оценка моделей на тестовых данных
cnn_loss, cnn_accuracy = cnn_model.evaluate(test_padded, test_labels)
lstm_loss, lstm_accuracy = lstm_model.evaluate(test_padded, test_labels)
gru_loss, gru_accuracy = gru_model.evaluate(test_padded, test_labels)

print(f'CNN Model - Loss: {cnn_loss}, Accuracy: {cnn_accuracy}')
print(f'LSTM Model - Loss: {lstm_loss}, Accuracy: {lstm_accuracy}')
print(f'GRU Model - Loss: {gru_loss}, Accuracy: {gru_accuracy}')

# Построение графиков точности
plt.plot(cnn_history.history['accuracy'], label='CNN Train Accuracy')
plt.plot(cnn_history.history['val_accuracy'], label='CNN Val Accuracy')
plt.plot(lstm_history.history['accuracy'], label='LSTM Train Accuracy')
plt.plot(lstm_history.history['val_accuracy'], label='LSTM Val Accuracy')
plt.plot(gru_history.history['accuracy'], label='GRU Train Accuracy')
plt.plot(gru_history.history['val_accuracy'], label='GRU Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
