import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# Título principal
st.title("📈 Regresión Lineal Simple - Aprendizaje Supervisado Hugo Rubio 741974")

st.write(
    "Esta aplicación permite cargar datos, entrenar un modelo de regresión lineal simple, "
    "calcular el coeficiente de determinación R² y visualizar la línea de regresión."
)

# 1️⃣ Cargar datos
st.header("1️⃣ Cargar datos")
file = st.file_uploader("Sube un archivo CSV con tus datos", type=["csv"])

if file is not None:
    data = pd.read_csv(file)
    st.write("Vista previa de los datos:")
    st.dataframe(data.head())

    columnas = data.columns.tolist()

    # Seleccionar variables
    st.subheader("Selecciona las variables")
    x_col = st.selectbox("Variable independiente (X)", columnas)
    y_col = st.selectbox("Variable dependiente (Y)", columnas)

    if x_col and y_col:
        X = data[[x_col]]
        y = data[y_col]

        # Entrenar modelo
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        modelo = LinearRegression()
        modelo.fit(X_train, y_train)

        st.success("✅ Modelo entrenado correctamente")

        # Mostrar ecuación
        st.write(
            f"### Ecuación del modelo: \n\n"
            f"**Y = {modelo.coef_[0]:.2f}X + {modelo.intercept_:.2f}**"
        )

        # Calcular R² y error
        y_pred = modelo.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        st.markdown("### 🔹 Mostrar el R²")
        st.write(f"**Coeficiente de determinación (R²):** {r2:.4f}")
        st.write(f"**Error cuadrático medio (MSE):** {mse:.4f}")

        # 2️⃣ Predicción manual
        st.header("2️⃣ Realiza una predicción")
        nuevo_valor = st.number_input(f"Ingrese un valor para {x_col}:", value=0.0)
        if st.button("Predecir"):
            prediccion = modelo.predict([[nuevo_valor]])[0]
            st.success(f"📊 Predicción para {x_col} = {nuevo_valor}: **{y_col} ≈ {prediccion:.2f}**")

            # 3️⃣ Visualización
            st.header("3️⃣ Visualización del modelo")
            fig, ax = plt.subplots()

            # Datos reales
            ax.scatter(X, y, color="blue", label="Datos reales")
            # Línea de regresión
            ax.plot(X, modelo.predict(X), color="red", label="Línea de regresión")
            # Punto de predicción
            ax.scatter(nuevo_valor, prediccion, color="green", s=100, label="Predicción")

            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.legend()
            st.pyplot(fig)
else:
    st.info("Por favor, sube un archivo CSV para comenzar.")
