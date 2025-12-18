import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix
)
from sklearn.model_selection import train_test_split

def load_data(filename):
    data = pd.read_csv(filename, sep=';', header=None)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values.astype(int)
    return X, y

def main():
    filename = 'dataset_processed.ssv'
    test_size = 0.2
    random_state = 42
    max_depth = 99
    min_samples = 5
    
    X, y = load_data(filename)
    print(f"Данные загружены: {X.shape[0]} примеров, {X.shape[1]} признаков")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Обучающая выборка: {X_train.shape[0]} примеров")
    print(f"Тестовая выборка:  {X_test.shape[0]} примеров")
    
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=random_state,
        min_samples_leaf=min_samples,
        criterion='entropy'
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary', pos_label=1)
    recall = recall_score(y_test, y_pred, average='binary', pos_label=1)
    f1 = f1_score(y_test, y_pred, average='binary', pos_label=1)
    
    print(f"\nМетрики качества:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-score:  {f1:.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\nConfusion Matrix:")
    print(f"  True Negative:  {tn}")
    print(f"  False Positive: {fp}")
    print(f"  False Negative: {fn}")
    print(f"  True Positive:  {tp}")
    
if __name__ == "__main__":
    main()