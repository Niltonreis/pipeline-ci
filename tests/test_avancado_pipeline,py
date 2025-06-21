import pandas as pd
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import train_test_split

from funcoes import pipeline

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

def test_no_nulls_in_data():
    assert not X_train.isnull().any().any(), "X_train possui valores nulos"
    assert not X_test.isnull().any().any(), "X_test possui valores nulos"
    assert not y_train.isnull().any().any(), "y_train possui valores nulos"
    assert not y_test.isnull().any().any(), "y_test possui valores nulos"


def test_model_performance():
    pipe = pipeline.create_pipeline(X_train)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    assert acc > 0.75, f"Acurácia abaixo do esperado: {acc:.2f}"


def test_pipeline_runs_on_sample():
    pipe = pipeline.create_pipeline(X_train)
    X_sample = X_train.sample(10, random_state=42)
    y_sample = y_train.loc[X_sample.index]

    try:
        pipe.fit(X_sample, y_sample)
        preds = pipe.predict(X_sample)
    except Exception as e:
        assert False, f"Pipeline falhou ao rodar em amostra: {e}"



def test_reproducibility():
    pipe1 = pipeline.create_pipeline(X_train)
    pipe2 = pipeline.create_pipeline(X_train)

    pipe1.fit(X_train, y_train)
    pipe2.fit(X_train, y_train)

    pred1 = pipe1.predict(X_test)
    pred2 = pipe2.predict(X_test)

    assert np.array_equal(pred1, pred2), "Predições diferentes entre execuções"


def test_expected_columns():
    expected_cols = ['education', 'age', 'fnlwgt']
    for col in expected_cols:
        assert col in X_train.columns, f"Coluna ausente no dataset: {col}"