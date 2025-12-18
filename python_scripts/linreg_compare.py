import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split


def load_diabetes_dataset():
    diabetes = pd.read_csv("diabetes.ssv", sep=';')
    X = diabetes.iloc[:, :-1]
    y = diabetes.iloc[:, -1]
    
    return X, y

def compare_models(X_train, y_train, X_test, y_test):
    
    print("\nSGDRegressor (градиентный спуск):")
    sgd = SGDRegressor(
        learning_rate='constant',
        eta0=0.01,
        max_iter=10000,
        tol=1e-6,
        random_state=42,
        alpha=0
    )
    sgd.fit(X_train, y_train)
    
    y_pred_sgd = sgd.predict(X_test)
    
    print(f"   MSE test:  {mean_squared_error(y_test, y_pred_sgd):.6f}")
    print(f"   R2 test:   {r2_score(y_test, y_pred_sgd):.6f}")
    print(f"   MAE test:  {mean_absolute_error(y_test, y_pred_sgd):.6f}")
        
    return sgd

def main():

    X, y = load_diabetes_dataset()
    
    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nРазделение данных:")
    print(f"  Обучающая выборка: {X_train.shape[0]} samples")
    print(f"  Тестовая выборка:  {X_test.shape[0]} samples")
    
    sgd = compare_models(X_train, y_train, X_test, y_test)
    

if __name__ == "__main__":
    main()