# salvar_modelo.py
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from funcoes import pipeline

df = pd.read_csv('adult.csv', na_values=['#NAME?'])
X = df.drop('income', axis=1)
y = df['income']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

pipe = pipeline.create_pipeline(X_train)
pipe.fit(X_train, y_train)

with open('model.pkl', 'wb') as f:
    pickle.dump(pipe, f)

print("Modelo salvo como model.pkl")