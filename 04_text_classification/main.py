from tensorflow.keras.preprocessing.text import Tokenizer
from keras.optimizers import RMSprop

from preprocessing.preprocessing import DataFrame, num_classes
from models.models import CNNModel #, GRUModel, LSTMModel


max_words = 20000
max_length = 500
num_classes = num_classes('data\\classes.txt')

tokenizer = Tokenizer(num_words=max_words)

train_df = DataFrame('data\\train.csv', tokenizer, 100, num_classes)
test_df = DataFrame('data\\test.csv', tokenizer, 100, num_classes)

X_train = train_df.df_padded
y_train = train_df.df_labels

X_test = test_df.df_padded
y_test = test_df.df_labels

cnn_model = CNNModel(max_words, num_classes)
cnn_model.compile(
    optimizer=RMSprop(learning_rate=5e-4),
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy']
)

epochs = 15
batch_size = 128

cnn_history = cnn_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
loss, accuracy = cnn_model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')
