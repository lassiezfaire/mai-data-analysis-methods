import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocessing(train_path: str, test_path: str, classes_path: str):
    # Преобразование данных в pandas DataFrame
    train_df = pd.read_csv(train_path, names=['Class', 'Title', 'Description'])
    test_df = pd.read_csv(test_path, names=['Class', 'Title', 'Description'])
    classes = pd.read_csv(classes_path, names=['Class'])

    pd.set_option('display.max_columns', None)

    # Объединение Title и Description в один текст
    train_df['Text'] = train_df['Title'] + ' ' + train_df['Description']
    test_df['Text'] = test_df['Title'] + ' ' + test_df['Description']

    # Преобразование классов в числовые значения
    label_encoder = LabelEncoder()
    train_df['Class'] = label_encoder.fit_transform(train_df['Class'])
    test_df['Class'] = label_encoder.transform(test_df['Class'])

    # Перемешиваем DataFrame
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    test_df = test_df.sample(frac=1).reset_index(drop=True)

    return train_df, test_df, classes
