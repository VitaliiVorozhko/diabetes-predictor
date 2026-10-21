from __future__ import annotations
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from joblib import load
from numpy.typing import ArrayLike

MODEL_FILE = "regression_model.joblib"

st.set_page_config(page_title="Прогноз прогресування діабету", page_icon="🩺", layout="wide")


@st.cache_resource
def get_model(filename: str = MODEL_FILE):
    """Завантажує модель; якщо файл відсутній (або несумісна версія sklearn) - донавчає її."""
    try:
        return load(filename)
    except Exception:
        from model import load_data, train_regression_model
        X, y = load_data()
        return train_regression_model(X, y)


@st.cache_data
def get_data():
    X = load("X.joblib")
    y = load("y.joblib")
    metrics = load("metrics.joblib") if os.path.exists("metrics.joblib") else {}
    return X, y, metrics


def load_and_predict(X: ArrayLike, filename: str = MODEL_FILE) -> np.ndarray:
    """Завантажує збережену модель і повертає прогноз для переданих даних."""
    model = get_model(filename)
    return model.predict(X)


def visualize_difference(age: float, bmi: float, bp: float, prediction: ArrayLike):
    """
    Графік: усі пацієнти (сірі точки), крива моделі по ІМТ при заданих віці й тиску,
    червона точка - прогноз для користувача, синя - найближчий реальний пацієнт,
    штрихова лінія - різниця між ними.
    """
    X, y, _ = get_data()
    pred_value = float(np.asarray(prediction).ravel()[0])

    Xv = X.to_numpy(dtype=float)
    scale = Xv.std(axis=0)
    dist = np.linalg.norm((Xv - np.array([age, bmi, bp])) / scale, axis=1)
    i = int(dist.argmin())
    actual_value = float(np.asarray(y)[i])
    actual_bmi = float(Xv[i, 1])
    difference = actual_value - pred_value

    grid = np.linspace(X["bmi"].min(), X["bmi"].max(), 100)
    curve_df = pd.DataFrame({"age": age, "bmi": grid, "bp": bp})
    curve = load_and_predict(curve_df)

    bg, fg = "#0E1117", "#FAFAFA"
    fig, ax = plt.subplots(figsize=(8, 4.8), facecolor=bg)
    ax.set_facecolor(bg)
    ax.scatter(X["bmi"], y, color="grey", alpha=0.45, label="Пацієнти (факт)")
    ax.plot(grid, curve, color="#2ECC71", linewidth=2.5,
            label=f"Модель (вік {age:.0f}, тиск {bp:.0f})")
    ax.scatter(actual_bmi, actual_value, color="#3B82F6", s=70, zorder=3, label="Найближчий пацієнт")
    ax.scatter(bmi, pred_value, color="#FF4B4B", s=110, zorder=4, label=f"Ваш прогноз: {pred_value:.0f}")
    ax.plot([actual_bmi, bmi], [actual_value, pred_value], "w--", linewidth=1)
    ax.annotate(f"Різниця: {difference:+.1f}",
                xy=((actual_bmi + bmi) / 2, (actual_value + pred_value) / 2),
                xytext=(10, 10), textcoords="offset points", color=fg)
    ax.set_title("Індекс прогресування хвороби залежно від ІМТ", color=fg)
    ax.set_xlabel("ІМТ (BMI)", color=fg)
    ax.set_ylabel("Прогресування хвороби (індекс)", color=fg)
    ax.tick_params(colors=fg)
    for s in ax.spines.values():
        s.set_color("#555")
    ax.grid(True, alpha=0.2)
    ax.legend(facecolor="#1C1F26", edgecolor="#555", labelcolor=fg)
    fig.tight_layout()
    st.pyplot(fig)


def create_streamlit_app():
    X, y, metrics = get_data()

    st.title("🩺 Прогноз прогресування діабету")
    st.write(
        "Застосунок оцінює, наскільки швидко прогресуватиме захворювання через рік, "
        "за віком, індексом маси тіла (ІМТ) і середнім артеріальним тиском. "
        "Дані: класичний датасет *diabetes* (442 пацієнти), модель: поліноміальна Ridge-регресія."
    )

    st.sidebar.header("Вхідні параметри")
    age = st.sidebar.slider("Вік (роки)", int(X["age"].min()), int(X["age"].max()), 45)
    bmi = st.sidebar.slider("ІМТ (BMI)", float(round(X["bmi"].min(), 1)), float(round(X["bmi"].max(), 1)), 26.0, step=0.1)
    bp = st.sidebar.slider("Середній артеріальний тиск (мм рт. ст.)", int(X["bp"].min()), int(X["bp"].max()), 94)

    if st.sidebar.button("Розрахувати прогноз"):
        st.session_state["predicted"] = True

    if not st.session_state.get("predicted"):
        st.info("Виберіть параметри на бічній панелі ліворуч та натисніть 'Розрахувати прогноз'.")
        return

    prediction = load_and_predict(pd.DataFrame({"age": [age], "bmi": [bmi], "bp": [bp]}))
    value = float(prediction[0])
    mean_y = float(np.mean(y))
    percentile = float((np.asarray(y) < value).mean() * 100)

    c1, c2, c3 = st.columns(3)
    c1.metric("Прогнозований індекс", f"{value:.1f}", f"{value - mean_y:+.1f} до середнього ({mean_y:.0f})")
    c2.metric("Вище, ніж у % пацієнтів", f"{percentile:.0f}%")
    if metrics:
        c3.metric("Якість моделі (R²)", f"{metrics['r2']:.2f}",
                  f"{metrics['r2'] - metrics.get('baseline_r2', metrics['r2']):+.2f} проти лінійної")

    visualize_difference(age, bmi, bp, prediction)

   


if __name__ == "__main__":
    create_streamlit_app()
