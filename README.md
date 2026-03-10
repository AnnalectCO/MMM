# 📊 MMM Studio – Marketing Mix Modeling en Streamlit

[Running in... ](https://mixmedia.streamlit.app/)
> Aplicación interactiva para construir modelos de **Marketing Mix Modeling (MMM)** a partir de cualquier dataset tabular, usando transformaciones Adstock v3, Hill y regresión OLS.


![Python](https://img.shields.io/badge/Python-3.10+-pink)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red)
![MMM](https://img.shields.io/badge/MMM-Annalect-blue)

---

## 🚀 Inicio Rápido

### 1. Clonar / descomprimir el repositorio

```bash
git clone https://github.com/tu_usuario/mmm-studio.git
cd mmm-studio
```

### 2. Crear entorno virtual (recomendado)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate.bat       # Windows
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Ejecutar la app

```bash
streamlit run app.py
```

Abre tu navegador en `http://localhost:8501`

---

## 🗂️ Estructura del Proyecto

```
mmm-studio/
├── app.py              ← Aplicación principal Streamlit
├── requirements.txt    ← Dependencias Python
├── sample_data/
│   └── sample_mmm.csv  ← Dataset de ejemplo (opcional)
└── README.md
```

---

## 📋 Flujo de Trabajo

La aplicación está organizada en **5 pestañas** que siguen el pipeline completo de un MMM:

### 1️⃣ Datos
- Carga archivos **CSV o Excel**
- Selecciona columna de **fecha** y **variable objetivo** (ventas)
- Explora estadísticas descriptivas y visualiza series temporales

### 2️⃣ Adstock
Aplica la transformación **Adstock v3** sobre variables de inversión/GRPs.

| Parámetro | Descripción |
|-----------|-------------|
| `fdecayRate` | Tasa de decaimiento (0–1). Ej: `0.5` |
| `peak` | Semana donde ocurre el máximo impacto. Ej: `1` |
| `length` | Duración total del efecto. Ej: `82` |

Ejemplo de configuración típica:
```python
'inversion_total': {'fdecayRate': 0.5, 'peak': 1, 'length': 82}
```

### 3️⃣ Hill
Aplica la **curva Hill / S-curve** para capturar rendimientos decrecientes.

| Parámetro | Descripción |
|-----------|-------------|
| `rho` | Punto de inflexión (media de X como valor inicial) |
| `p` | Forma de la curva (1 = Michaelis-Menten) |
| `beta` | Escala del efecto máximo |
| `alpha` | Intercepto |

### 4️⃣ Rezagos y Diferencias
- Genera **lags** (rezagos): `col_lag1`, `col_lag2`, etc.
- Genera **diferencias**: `col_d1`, etc.
- Crea columnas combinadas (sumas de inversiones)
- Filtra el período de modelado

### 5️⃣ Modelo
- Selecciona variables predictoras
- Define **restricciones de contribución** por tipo de variable:

| Tipo de variable | Rango objetivo |
|-----------------|----------------|
| Inversión propia | 7% – 12% |
| Competencia / IBOPE | 5% – 9% |
| Quincena, navidad, promos | < 5% |

- Visualiza semáforo de contribuciones (🟢 dentro / 🔴 fuera del rango)
- Verifica **R² ≥ 0.80**
- Diagnósticos de residuales (histograma, Q-Q, fitted vs residuals)
- Exporta coeficientes y contribuciones a CSV

---

## 📐 Criterios de Optimización del Modelo

El modelo se considera válido cuando cumple:

| Criterio | Umbral |
|----------|--------|
| R² | ≥ 0.80 |
| Contribución inversión propia | 7% – 12% |
| Contribución competencia/IBOPE | 5% – 9% |
| Variables estacionales/promo | < 5% c/u |

---

## 🧮 Funciones Principales

### `adstockv3_v1(afGRPs, fdecayRate, peak, length)`
Transformación Adstock con peak retardado y longitud de efecto controlada.

### `hill(X, rho, p, beta, alpha)`
Curva de saturación Hill para capturar rendimientos decrecientes de la inversión.

### `ajustar_ols(df, target_col, x_cols)`
Ajusta regresión OLS vía `statsmodels` y calcula contribuciones relativas (%).

---

## 📦 Dependencias

| Paquete | Versión mínima |
|---------|---------------|
| streamlit | 1.35 |
| pandas | 2.0 |
| numpy | 1.24 |
| statsmodels | 0.14 |
| matplotlib | 3.7 |
| openpyxl | 3.1 |

---

## 🔧 Configuración Avanzada

Para personalizar colores, rango de fechas por defecto, o agregar más transformaciones, edita directamente `app.py`. Las funciones core están al inicio del archivo claramente comentadas.

---

<img width="409" height="246" alt="image" src="https://github.com/user-attachments/assets/db0fa8a4-77f2-46e7-9113-39d48c2744af" />

## 📄 Licencia

MIT – libre uso y modificación.
