import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix
)
from sklearn.model_selection import train_test_split

def load_processed_data(filename):
    
    train_data = pd.read_csv(filename, sep=';', header=None)
    X = train_data.iloc[:, :-1].values
    y = train_data.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X,y)
    
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    
    print(f"Обучающая выборка: {X_train.shape[0]} примеров, {X_train.shape[1]} признаков")
    print(f"Тестовая выборка:  {X_test.shape[0]} примеров, {X_test.shape[1]} признаков")
    
    return X_train, X_test, y_train, y_test

def train(X_train, X_test, y_train, y_test):
    model = LogisticRegression(
        solver='liblinear',
        max_iter=5000,
        random_state=42,
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    precision = precision_score(y_test, y_pred, average='binary', pos_label=1)
    recall = recall_score(y_test, y_pred, average='binary', pos_label=1)
    f1 = f1_score(y_test, y_pred, average='binary', pos_label=1)
    
    print(f"Accuracy:               {accuracy:.4f}")
    print(f"Precision:      {precision:.4f}")
    print(f"Recall:         {recall:.4f}")
    print(f"F1-score:       {f1:.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    
    tn, fp, fn, tp = cm.ravel()
    print(f"\nTrue Negative:  {tn}")
    print(f"False Positive: {fp}")
    print(f"False Negative: {fn}")
    print(f"True Positive:  {tp}")
    
    return model, y_pred

def main():
    filename = 'dataset_processed.ssv'
    
    X_train, X_test, y_train, y_test = load_processed_data(filename)
    
    model, y_pred = train(X_train, X_test, y_train, y_test)
            

if __name__ == "__main__":
    main()