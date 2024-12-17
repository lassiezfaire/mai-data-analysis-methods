from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras import callbacks

from input_output.preprocessing import preprocessing, num_classes
from input_output.graphs import plot_acc_loss
from models.models import CNNModel, LSTMModel, GRUModel

num_classes = num_classes(filepath='data\\classes.txt')

X_train, y_train = preprocessing(csv_name='train.csv', clean_csv_name='clean_train.csv',
                                 path_to_data='data', num_classes=num_classes)
X_test, y_test = preprocessing(csv_name='test.csv', clean_csv_name='clean_test.csv',
                               path_to_data='data', num_classes=num_classes)

tokenizer = Tokenizer(num_words=100_000)
tokenizer.fit_on_texts(X_train)

vocab_size = len(tokenizer.index_word)

X_train_token = tokenizer.texts_to_sequences(X_train)
X_test_token = tokenizer.texts_to_sequences(X_test)

sequence_len = 50
X_train_token = pad_sequences(X_train_token, maxlen=sequence_len)
X_test_token = pad_sequences(X_test_token, maxlen=sequence_len)

early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

cnn_model = CNNModel(vocab_size=vocab_size, sequence_len=sequence_len, num_classes=num_classes)
ccn_history = cnn_model.fit(X_train_token, y_train, epochs=20, batch_size=32, validation_split=0.2,
                            callbacks=[early_stopping])
loss, acc = cnn_model.evaluate(X_test_token, y_test, verbose=1)

print(f'CNN Model - Loss: {loss:.4f}, Accuracy: {acc:.4f}%')

plot_acc_loss(ccn_history, 'accuracy', acc, cnn_model.__name__)
plot_acc_loss(ccn_history, 'loss', loss, cnn_model.__name__)

lstm_model = LSTMModel(vocab_size=vocab_size, sequence_len=sequence_len, num_classes=num_classes)
lstm_history = lstm_model.fit(X_train_token, y_train, epochs=20, batch_size=32, validation_split=0.2,
                              callbacks=[early_stopping])
loss, acc = lstm_model.evaluate(X_test_token, y_test, verbose=1)

print(f'LSTM Model - Loss: {loss:.4f}, Accuracy: {acc:.4f}%')

plot_acc_loss(lstm_history, 'accuracy', acc, lstm_model.__name__)
plot_acc_loss(lstm_history, 'loss', loss, lstm_model.__name__)

gru_model = GRUModel(vocab_size=vocab_size, sequence_len=sequence_len, num_classes=num_classes)
gru_history = gru_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2,
                            callbacks=[early_stopping])
loss, acc = gru_model.evaluate(X_test_token, y_test, verbose=1)

print(f'GRU Model - Loss: {loss:.4f}, Accuracy: {acc:.4f}%')

plot_acc_loss(gru_history, 'accuracy', acc, gru_model.__name__)
plot_acc_loss(gru_history, 'loss', loss, gru_model.__name__)
