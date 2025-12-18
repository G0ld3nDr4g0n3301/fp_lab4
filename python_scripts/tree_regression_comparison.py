import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

def load_diabetes_dataset():
    diabetes = pd.read_csv("diabetes.ssv", sep=';', header=None)
    X = diabetes.iloc[:, :-1].values
    y = diabetes.iloc[:, -1].values
    
    return X, y

def compare_models(X_train, y_train, X_test, y_test, max_depth):
    print("\nDecisionTreeRegressor (дерево решений):")
    dtr = DecisionTreeRegressor(
        max_depth=max_depth,
        random_state=42,
        criterion='squared_error',
        min_samples_leaf=10
    )
    dtr.fit(X_train, y_train)
    
    y_pred_dtr = dtr.predict(X_test)
    
    print(f"   MSE test:  {mean_squared_error(y_test, y_pred_dtr):.6f}")
    print(f"   R2 test:   {r2_score(y_test, y_pred_dtr):.6f}")
    print(f"   MAE test:  {mean_absolute_error(y_test, y_pred_dtr):.6f}")
    
    print(f"   Глубина дерева: {dtr.get_depth()}")
    print(f"   Количество листьев: {dtr.get_n_leaves()}")
    
    return dtr, y_pred_dtr

def main():
    test_size = 0.2
    random_state = 42
    max_depth = 20
    min_samples = 5
    
    X, y = load_diabetes_dataset()
    
    print(f"Датасет diabetes:")
    print(f"  Образцов: {X.shape[0]}, признаков: {X.shape[1]}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"\nРазделение данных:")
    print(f"  Обучающая выборка: {X_train.shape[0]} образцов")
    print(f"  Тестовая выборка:  {X_test.shape[0]} образцов")
    
    dtr, y_pred = compare_models(X_train, y_train, X_test, y_test, max_depth)

if __name__ == "__main__":
    main()