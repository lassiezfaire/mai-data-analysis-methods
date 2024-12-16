from keras import layers
from keras import models
from keras import regularizers

class CNNModel(models.Sequential):
    def __init__(self, vocab_size, sequence_len, num_classes):
        super(CNNModel, self).__init__()
        self.add(layers.Embedding(vocab_size, 128, input_length=sequence_len))
        self.add(layers.Conv1D(128, 5, activation='relu'))
        self.add(layers.MaxPooling1D(5))
        self.add(layers.Dropout(0.7))
        self.add(layers.GlobalMaxPooling1D())
        self.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
        self.add(layers.Dropout(0.7))
        self.add(layers.Dense(num_classes, activation='softmax'))

        self.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

class LSTMModel(models.Sequential):
    def __init__(self, vocab_size, sequence_len, num_classes):
        super(LSTMModel, self).__init__()
        self.add(layers.Embedding(vocab_size, 128, input_length=sequence_len))
        self.add(layers.LSTM(128, return_sequences=True))
        self.add(layers.LSTM(128))
        self.add(layers.Dropout(0.7))
        self.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
        self.add(layers.Dropout(0.7))
        self.add(layers.Dense(num_classes, activation='softmax'))

        self.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

class GRUModel(models.Sequential):
    def __init__(self, vocab_size, sequence_len, num_classes):
        super(GRUModel, self).__init__()
        self.add(layers.Embedding(vocab_size, 128, input_length=sequence_len))
        self.add(layers.GRU(128, return_sequences=True))
        self.add(layers.GRU(128))
        self.add(layers.Dropout(0.7))
        self.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
        self.add(layers.Dropout(0.7))
        self.add(layers.Dense(num_classes, activation='softmax'))

        self.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
