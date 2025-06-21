import pickle
from sklearn.metrics import accuracy_score

def test_model_pickle_accuracy():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    # X_test e y_test precisam estar disponíveis (carregue ou use fixture)
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv('adult.csv', na_values=['#NAME?'])
    X = df.drop('income', axis=1)
    y = df['income']
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    assert acc > 0.75, f"Acurácia do modelo salvo está baixa: {acc:.2f}"