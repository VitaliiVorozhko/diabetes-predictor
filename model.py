from __future__ import annotations
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from joblib import dump
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# Ознаки, які користувач задає у застосунку (в реальних одиницях виміру)
FEATURES = ["age", "bmi", "bp"]
MODEL_FILE = "regression_model.joblib"


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """
    Завантажує датасет diabetes (вбудований у scikit-learn, інтернет не потрібен).

    scaled=False повертає ознаки в реальних одиницях (вік у роках, ІМТ, тиск),
    а не в стандартизованому вигляді, тому слайдери в застосунку зрозумілі людині.
    Ціль (target) - індекс прогресування хвороби через рік після базового виміру.
    """
    data = load_diabetes(scaled=False, as_frame=True)
    X = data.data[FEATURES]
    y = data.target
    return X, y


def train_regression_model(X_train: ArrayLike, y_train: ArrayLike, degree: int = 2) -> Pipeline:
    """
    Навчає поліноміальну регресію з регуляризацією (Ridge).

    Pipeline: масштабування -> поліноміальні ознаки -> Ridge.
    Масштабування потрібне, щоб ознаки в різних одиницях (роки, ІМТ, мм рт. ст.)
    не домінували одна над одною; Ridge стримує перенавчання на малій вибірці (442 рядки).
    """
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("ridge", Ridge(alpha=10.0)),
    ])
    model.fit(X_train, y_train)
    return model


def save_regression_model(model: Pipeline, filename: str = MODEL_FILE):
    """Зберігає навчений pipeline на диск (joblib)."""
    dump(model, filename)


def evaluate_regression_model(model: Pipeline, X_test: ArrayLike, y_test: ArrayLike,
                              X_train: ArrayLike | None = None, y_train: ArrayLike | None = None) -> dict:
    """
    Рахує MSE та R^2 на тестових даних, друкує їх і повертає словником.
    Якщо передано train-дані, додатково порівнює з простою лінійною регресією (baseline).
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"R^2: {r2}")
    metrics = {"mse": float(mse), "r2": float(r2)}

    if X_train is not None and y_train is not None:
        baseline = LinearRegression().fit(X_train, y_train)
        base_pred = baseline.predict(X_test)
        metrics["baseline_mse"] = float(mean_squared_error(y_test, base_pred))
        metrics["baseline_r2"] = float(r2_score(y_test, base_pred))
        print(f"Baseline (linear) MSE: {metrics['baseline_mse']}, R^2: {metrics['baseline_r2']}")
    return metrics


def save_initial_datasets(X: ArrayLike, y: ArrayLike, metrics: dict | None = None):
    """Зберігає повні X та y (для графіка) і метрики моделі (для відображення в застосунку)."""
    dump(X, "X.joblib")
    dump(y, "y.joblib")
    if metrics is not None:
        dump(metrics, "metrics.joblib")


if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = train_regression_model(X_train, y_train)
    metrics = evaluate_regression_model(model, X_test, y_test, X_train, y_train)

    save_regression_model(model)
    save_initial_datasets(X, y, metrics)
