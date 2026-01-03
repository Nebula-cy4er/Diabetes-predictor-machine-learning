#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[5]:


pip install xgboost


# In[14]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.impute import SimpleImputer
import xgboost as xgb
import matplotlib.pyplot as plt
import pickle

def preprocess_data(df):
    cols_with_missing = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols_with_missing] = df[cols_with_missing].replace(0, np.nan)

    imputer = SimpleImputer(strategy='median')
    df[cols_with_missing] = imputer.fit_transform(df[cols_with_missing])
    
    return df

def main():
    try:
        df = pd.read_csv("C:/Users/vaish/OneDrive/Documents/6th Sem/health predictor/diabetes.csv")
    except FileNotFoundError:
        print("Error: 'diabetes.csv' not found.")
        return
    
    df = preprocess_data(df)
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)


    print("\nClassification Report:\n", report)
    print("Accuracy:", accuracy)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap='Purples')
    plt.title("Confusion Matrix")
    plt.grid(False)
    plt.show()

   
    xgb.plot_importance(model)
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.show()


    with open("diabetes_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

if __name__ == "__main__":
    main()


# In[ ]:




