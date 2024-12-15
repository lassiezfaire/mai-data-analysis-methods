from keras import Model
from keras.layers import Embedding, Dense, Dropout, Conv1D, MaxPooling1D, GlobalMaxPooling1D

class CNNModel(Model):
    def __init__(self, max_words: int, num_classes: int):
        super().__init__()

        self.max_words = max_words
        self.num_classes = num_classes

        self.embedding_layer = Embedding(self.max_words, 50)
        self.conv1 = Conv1D(256, 3, activation='relu', padding='same')
        self.pool1 = MaxPooling1D(2)
        self.dropout1 = Dropout(0.5)
        self.conv2 = Conv1D(256, 3, activation='relu', padding='same')
        self.pool2 = MaxPooling1D(2)
        self.dropout2 = Dropout(0.5)
        self.conv3 = Conv1D(256, 3, activation='relu', padding='same')
        self.pool3 = MaxPooling1D(2)
        self.dropout3 = Dropout(0.5)
        self.conv4 = Conv1D(256, 3, activation='relu', padding='same')
        self.pool4 = MaxPooling1D(2)
        self.dropout4 = Dropout(0.5)
        self.conv5 = Conv1D(256, 3, activation='relu', padding='same')
        self.pool5 = MaxPooling1D(2)
        self.dropout5 = Dropout(0.5)
        self.global_pool = GlobalMaxPooling1D()
        self.dense = Dense(256, activation='relu')
        self.output_layer = Dense(self.num_classes, activation='softmax')

    def call(self, inputs):
        x = self.embedding_layer(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.pool4(x)
        x = self.conv5(x)
        x = self.pool5(x)
        x = self.global_pool(x)
        x = self.dense(x)
        return self.output_layer(x)

# class GRUModel(BaseModel):
#     def __init__(self, num_classes: int, max_length: int):
#         super().__init__()
#
#         self.gru_model = Sequential([
#             Embedding(input_dim=20000, output_dim=128, input_length=max_length),
#             GRU(128, return_sequences=True),
#             GRU(128),
#             Dense(128, activation='relu'),
#             Dropout(0.5),
#             Dense(num_classes, activation='softmax')
#         ])
#
#         self.gru_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#         self.gru_model.summary()
#
# class LSTMModel(BaseModel):
#     def __init__(self, num_classes: int, max_length: int):
#         super().__init__()
#
#         self.lstm_model = Sequential([
#             Embedding(input_dim=20000, output_dim=128, input_length=max_length),
#             LSTM(128, return_sequences=True),
#             LSTM(128),
#             Dense(128, activation='relu'),
#             Dropout(0.5),
#             Dense(num_classes, activation='softmax')
#         ])
#
#         self.lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#         self.lstm_model.summary()
