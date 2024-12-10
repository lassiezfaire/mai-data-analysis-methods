from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Dropout, LSTM

def lstm_model(num_classes: int, max_length: int):

    lstm_model = Sequential([
        Embedding(input_dim=20000, output_dim=128, input_length=max_length),
        LSTM(128, return_sequences=True),
        LSTM(128),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    lstm_model.summary()
    return lstm_model