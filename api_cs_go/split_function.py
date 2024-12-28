import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def split_func(dataframe):
    dataframe = dataframe[[col for col in dataframe.columns if 'ban' not in col.lower()]]

    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(dataframe)

    scaled = pd.DataFrame(scaled_data, columns=dataframe.columns)

    X = scaled.drop(columns=['y'], axis=1)
    y = dataframe['y']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if isinstance(X_train, (pd.DataFrame, pd.Series)):
        X_train = X_train.values
        X_test = X_test.values
        y_train = y_train.values
        y_test = y_test.values

    # тензори
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    return X_train, X_test, y_train, y_test
