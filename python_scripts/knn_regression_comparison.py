import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

def load_diabetes_dataset():
    diabetes = pd.read_csv("diabetes.ssv", sep=';', header=None)
    X = diabetes.iloc[:, :-1].values
    y = diabetes.iloc[:, -1].values
    return X, y

def compare_knn_regression(X_train, y_train, X_test, y_test, k, weights):
    print(f"\nKNeighborsRegressor (k={k}, weights='{weights}'):")
    
    knr = KNeighborsRegressor(
        n_neighbors=k,
        weights=weights,
        metric='euclidean'
    )
    knr.fit(X_train, y_train)
    
    y_pred = knr.predict(X_test)
    
    print(f"   MSE test:  {mean_squared_error(y_test, y_pred):.6f}")
    print(f"   R2 test:   {r2_score(y_test, y_pred):.6f}")
    print(f"   MAE test:  {mean_absolute_error(y_test, y_pred):.6f}")
    
    return knr, y_pred

def main():
    test_size = 0.2
    random_state = 42
    k_neighbors = 5
    weights_type = 'distance'
    
    X, y = load_diabetes_dataset()
    
    print(f"Датасет diabetes:")
    print(f"  Образцов: {X.shape[0]}, признаков: {X.shape[1]}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"\nРазделение данных:")
    print(f"  Обучающая выборка: {X_train.shape[0]} образцов")
    print(f"  Тестовая выборка:  {X_test.shape[0]} образцов")
    model, y_pred = compare_knn_regression(
        X_train, y_train, X_test, y_test, 
        k_neighbors, weights_type
    )

if __name__ == "__main__":
    main()