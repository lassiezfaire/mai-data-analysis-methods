from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, GRU, LSTM

class BaseModel:
    def __init__(self):
        pass

class CNNModel(BaseModel):
    def __init__(self, num_classes: int, max_length: int):
        super().__init__()

        self.cnn_model = Sequential([
            Embedding(input_dim=20000, output_dim=128, input_length=max_length),
            Conv1D(filters=128, kernel_size=5, activation='relu'),
            GlobalMaxPooling1D(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])

        self.cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.cnn_model.summary()

class GRUModel(BaseModel):
    def __init__(self, num_classes: int, max_length: int):
        super().__init__()

        self.gru_model = Sequential([
            Embedding(input_dim=20000, output_dim=128, input_length=max_length),
            GRU(128, return_sequences=True),
            GRU(128),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])

        self.gru_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.gru_model.summary()

class LSTMModel(BaseModel):
    def __init__(self, num_classes: int, max_length: int):
        super().__init__()

        self.lstm_model = Sequential([
            Embedding(input_dim=20000, output_dim=128, input_length=max_length),
            LSTM(128, return_sequences=True),
            LSTM(128),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])

        self.lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.lstm_model.summary()
