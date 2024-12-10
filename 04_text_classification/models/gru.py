from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Dropout, GRU

def gru_model(num_classes: int, max_length: int):

    gru_model = Sequential([
        Embedding(input_dim=20000, output_dim=128, input_length=max_length),
        GRU(128, return_sequences=True),
        GRU(128),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    gru_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    gru_model.summary()
    return gru_model