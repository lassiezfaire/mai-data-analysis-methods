from tensorflow.keras.preprocessing.text import Tokenizer

from preprocessing.preprocessing import DataFrame, num_classes
from models.models import CNNModel, GRUModel, LSTMModel

tokenizer = Tokenizer(num_words=20000)
max_length = 100
num_classes = num_classes('data\\classes.txt')

train_df = DataFrame('data\\train.csv', tokenizer, 100, num_classes)
test_df = DataFrame('data\\test.csv', tokenizer, 100, num_classes)

# print(train_df.df.head(5))
# print(test_df.df.head(5))

cnn_model = CNNModel(num_classes, max_length)
lstm_model = LSTMModel(num_classes, max_length)
gru_model = GRUModel(num_classes, max_length)

epochs = 15
batch_size = 128

cnn_history = cnn_model.cnn_model.fit(train_df.df_padded, train_df.df_labels, epochs=epochs,
                                      batch_size=batch_size, validation_split=0.2)
lstm_history = lstm_model.lstm_model.fit(train_df.df_padded, train_df.df_labels, epochs=epochs,
                                         batch_size=batch_size, validation_split=0.2)
gru_history = gru_model.gru_model.fit(train_df.df_padded, train_df.df_labels, epochs=epochs,
                                      batch_size=batch_size, validation_split=0.2)

# Оценка моделей на тестовых данных
cnn_loss, cnn_accuracy = cnn_model.cnn_model.evaluate(test_df.df_padded, test_df.df_labels)
lstm_loss, lstm_accuracy = lstm_model.lstm_model.evaluate(test_df.df_padded, test_df.df_labels)
gru_loss, gru_accuracy = gru_model.gru_model.evaluate(test_df.df_padded, test_df.df_labels)

print(f'CNN Model - Loss: {cnn_loss}, Accuracy: {cnn_accuracy}')
print(f'LSTM Model - Loss: {lstm_loss}, Accuracy: {lstm_accuracy}')
print(f'GRU Model - Loss: {gru_loss}, Accuracy: {gru_accuracy}')
