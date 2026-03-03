import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURACIÓN GENERAL
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AutoMMMate",
    layout="wide",
    page_icon="📊"
)

# ─────────────────────────────────────────────
# ESTILOS GLOBALES (UI PREMIUM)
# ─────────────────────────────────────────────
st.markdown("""
<style>

/* Reduce espacio superior */
.block-container {
    padding-top: 0.5rem;
}

/* Tabs más compactos */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
}

/* Sidebar oscuro moderno */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a, #111827);
}

[data-testid="stSidebar"] * {
    color: #e5e7eb !important;
}

/* Títulos con gradiente */
.section-title {
    font-size: 1.6rem;
    font-weight: 700;
    margin-top: 1.5rem;
    margin-bottom: 0.8rem;
    background: linear-gradient(90deg, #2563eb, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Caja tipo métrica personalizada */
.metric-box {
    background: #1e293b;
    border-radius: 12px;
    padding: 14px;
    border-left: 4px solid #2563eb;
    margin-bottom: 8px;
    color: white;
}

/* Ajustes generales */
h1, h2, h3 {
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HERO HEADER PRINCIPAL
# ─────────────────────────────────────────────
st.markdown("""
<div style="
    background: linear-gradient(135deg, #0f172a, #1e293b);
    padding: 2.2rem 2rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    box-shadow: 0 10px 25px rgba(0,0,0,0.25);
">
    <h1 style="
        color: white;
        font-size: 2.5rem;
        margin-bottom: 0.3rem;
        font-weight: 700;
    ">
        📊 MMM Studio
    </h1>
    <p style="
        color: #cbd5e1;
        font-size: 1.05rem;
        margin: 0;
    ">
        Marketing Mix Modeling · Modelado OLS · Análisis de Contribuciones
    </p>
</div>
""", unsafe_allow_html=True)
# ─────────────────────────────────────────────
#  FUNCIONES CORE
# ─────────────────────────────────────────────

def adstockv3_v1(afGRPs, fdecayRate, peak=1, length=600):
    """Adstock v3 – con peak y length configurables."""
    afGRPs_1 = np.array(afGRPs, dtype=float)
    if not (0 <= fdecayRate <= 1):
        raise ValueError("fdecayRate debe estar entre 0 y 1.")
    if peak < 1 or peak >= len(afGRPs_1):
        raise ValueError("peak debe ser >=1 y menor que la longitud de la serie.")
    if length <= peak:
        raise ValueError("length debe ser mayor que peak.")

    afAdStockedGRPs = np.zeros_like(afGRPs_1)
    for i in range(len(afGRPs_1)):
        value = afGRPs_1[i]
        if value > 0:
            tmp = np.zeros_like(afGRPs_1)
            tmp[i] = value
            tmp3 = np.roll(tmp, peak - 1)
            tmp3[:peak - 1] = 0
            res = []
            acc = 0
            for x in tmp3:
                acc = acc * (1 - fdecayRate) + x
                res.append(acc)
            res = np.array(res)
            if peak > 1:
                k = i
                for j in range(1, peak):
                    if (k + j) < len(afGRPs_1):
                        res[k + j] = j / peak * value
            if (i + length) < len(res):
                res[i + length:] = 0
            if res.sum() != 0:
                res *= value / res.sum()
            afAdStockedGRPs += res

    afAdStockedGRPs2 = []
    acc = 0
    for x in afGRPs_1:
        acc = acc * (1 - fdecayRate) + x
        afAdStockedGRPs2.append(acc)
    afAdStockedGRPs2 = np.array(afAdStockedGRPs2)
    if afAdStockedGRPs2.sum() != 0:
        afAdStockedGRPs2 *= afGRPs_1.sum() / afAdStockedGRPs2.sum()

    idx = afGRPs.index if isinstance(afGRPs, pd.Series) else None
    return pd.Series(afAdStockedGRPs, index=idx)


def hill(X, rho=None, p=1, beta=1, alpha=0):
    """Transformación Hill / S-curve."""
    X = np.asarray(X, dtype=float)
    if rho is None:
        rho = np.mean(X)
    denom = X**p + rho**p
    denom = np.where(denom == 0, np.finfo(float).eps, denom)
    result = alpha + beta * (X**p) / denom
    return result.item() if result.size == 1 else result


def ajustar_ols(df, target_col, x_cols):
    """Ajusta OLS y devuelve (modelo, contribuciones%)."""
    df_model = df[x_cols + [target_col]].dropna()
    X = sm.add_constant(df_model[x_cols])
    y = df_model[target_col]
    modelo = sm.OLS(y, X).fit()

    # Contribuciones relativas
    contri = {}
    for col in x_cols:
        contrib_abs = modelo.params.get(col, 0) * df_model[col]
        contri[col] = contrib_abs.sum() / y.sum() * 100
    return modelo, contri


def calcular_metricas(modelo):
    y_true = modelo.model.endog
    y_pred = modelo.fittedvalues

    # Evitar división por cero
    y_true_safe = np.where(y_true == 0, np.finfo(float).eps, y_true)
    mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100

    return {
        "R²": round(modelo.rsquared, 4),
        "Adj R²": round(modelo.rsquared_adj, 4),
        "MAPE": round(mape, 2),
    }


def plot_fit(df_rezagos, modelo, target_col, fecha_col):
    ventas = df_rezagos[target_col].reset_index(drop=True)
    pred   = modelo.fittedvalues.reset_index(drop=True)
    resid  = modelo.resid.reset_index(drop=True)
    fechas = pd.to_datetime(df_rezagos[fecha_col]).reset_index(drop=True)
    x      = np.arange(len(ventas))
    semanas = fechas.dt.isocalendar().week.astype(int)
    years   = fechas.dt.year
    unique_years   = years.unique()
    year_positions = [x[years == y][0] for y in unique_years]

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(x, ventas, lw=1.8, color="#1e293b", label="Ventas Reales")
    ax.plot(x, pred,   lw=1.5, color="#3b82f6", label="Ventas Pred", linestyle="--")
    ax.bar(x, resid, width=1.0, alpha=0.35, color="#94a3b8", label="Residuales")
    ax.set_ylabel("Ventas")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_xticks(year_positions)
    ax.set_xticklabels(unique_years, fontsize=12, fontweight="bold")
    N = max(1, len(x) // 20)
    indices = x[::N]
    labels  = semanas[::N]
    y_min, y_max = ax.get_ylim()
    offset_sem  = (y_max - y_min) * 0.05
    offset_year = (y_max - y_min) * 0.12
    for pos, lab in zip(indices, labels):
        ax.text(pos, y_min - offset_sem, f"S{lab}", ha="center", va="top", fontsize=7, rotation=45)
    for pos, year in zip(year_positions, unique_years):
        ax.text(pos, y_min - offset_year, str(year), ha="center", fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", which="major", pad=28)
    ax.legend(loc="upper center", ncol=3, frameon=False)
    fig.tight_layout()
    return fig


def plot_contrib(contri_dict, limites):
    """Gráfico de barras de contribuciones con bandas objetivo."""
    vars_ = list(contri_dict.keys())
    vals  = [contri_dict[v] for v in vars_]

    fig, ax = plt.subplots(figsize=(10, 0.5 * len(vars_) + 2))
    colors = []
    for v, val in zip(vars_, vals):
        lim = limites.get(v)
        if lim:
            lo, hi = lim
            colors.append("#22c55e" if lo <= val <= hi else "#ef4444")
        else:
            colors.append("#64748b")
    bars = ax.barh(vars_, vals, color=colors, edgecolor="white", height=0.6)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=9)
    for v, lim in limites.items():
        if v in vars_:
            idx = vars_.index(v)
            ax.barh(idx, lim[1] - lim[0], left=lim[0], height=0.6,
                    color="black", alpha=0.12, zorder=0)
    ax.set_xlabel("Contribución (%)")
    ax.axvline(0, color="black", lw=0.8)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────
# SESSION STATE DEFAULTS (ÚNICO BLOQUE)
# ─────────────────────────────────────────────

defaults = dict(
    df_raw=None,
    df_adstock=None,
    df_hill=None,
    df_rezagos=None,
    modelo=None,
    contri=None,
    adstock_params={},
    hill_params={},
    lag_params={},
    diff_params={},
    target_col=None,
    fecha_col=None,
    x_cols=[]
)

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("📊 MMM Studio")
    st.caption("Marketing Mix Modeling")
    st.divider()
    uploaded = st.file_uploader("📁 Cargar Dataset", type=["csv", "xlsx"])
    if uploaded:
        try:
            if uploaded.name.endswith(".csv"):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)
            st.session_state.df_raw = df
            st.success(f"✅ {df.shape[0]} filas × {df.shape[1]} cols")
        except Exception as e:
            st.error(f"Error al leer archivo: {e}")

    if st.session_state.df_raw is not None:
        cols = st.session_state.df_raw.columns.tolist()
        st.session_state.fecha_col = st.selectbox("📅 Columna de Fecha", cols)
        st.session_state.target_col = st.selectbox("🎯 Variable Objetivo (ventas)", cols,
                                                     index=min(1, len(cols)-1))

# ─────────────────────────────────────────────
#  TABS PRINCIPALES
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1️⃣ Datos", "2️⃣ Adstock", "3️⃣ Hill", "4️⃣ Rezagos/Difs", "5️⃣ Modelo"
])

# ─────────────────────────────────────────────
# SECCIÓN MODELO
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">Modelado OLS – MMM</div>', unsafe_allow_html=True)

# 🔹 Métricas placeholder (luego puedes conectarlas al modelo real)
col1, col2, col3 = st.columns(3)

if st.session_state.modelo is not None:
    metricas = calcular_metricas(st.session_state.modelo)

    col1.metric("R²", metricas["R²"])
    col2.metric("Adj R²", metricas["Adj R²"])
    col3.metric("MAPE", f'{metricas["MAPE"]}%')
else:
    col1.metric("R²", "—")
    col2.metric("Adj R²", "—")
    col3.metric("MAPE", "—")
                

# ══════════════════════════════════════════════
#  TAB 1 – EXPLORACIÓN DE DATOS
# ══════════════════════════════════════════════
with tab1:
    st.header("Exploración de Datos")
    if st.session_state.df_raw is None:
        st.info("⬅️ Carga un dataset desde la barra lateral para comenzar.")
    else:
        df = st.session_state.df_raw
        c1, c2, c3 = st.columns(3)
        c1.metric("Filas", df.shape[0])
        c2.metric("Columnas", df.shape[1])
        c3.metric("Valores nulos", int(df.isnull().sum().sum()))

        st.subheader("Vista previa")
        st.dataframe(df.head(50), use_container_width=True)

        st.subheader("Estadísticas descriptivas")
        st.dataframe(df.describe(), use_container_width=True)

        # Gráficas rápidas
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if num_cols:
            sel = st.multiselect("Visualizar series", num_cols, default=num_cols[:3])
            if sel and st.session_state.fecha_col:
                fig, ax = plt.subplots(figsize=(14, 4))
                for c in sel:
                    ax.plot(df[st.session_state.fecha_col], df[c], label=c, lw=1.5)
                ax.legend(ncol=4, frameon=False, fontsize=8)
                ax.grid(True, linestyle="--", alpha=0.3)
                plt.xticks(rotation=45)
                fig.tight_layout()
                st.pyplot(fig)

# ══════════════════════════════════════════════
#  TAB 2 – ADSTOCK
# ══════════════════════════════════════════════
with tab2:
    st.header("Transformación Adstock v3")
    if st.session_state.df_raw is None:
        st.info("Carga primero un dataset.")
    else:
        df = st.session_state.df_raw.copy()
        num_cols = df.select_dtypes(include=np.number).columns.tolist()

        st.markdown("Selecciona las variables a transformar y configura sus parámetros Adstock.")
        cols_adstock = st.multiselect("Variables para Adstock", num_cols)

        adstock_params = {}
        if cols_adstock:
            st.markdown("### Parámetros por variable")
            grid = st.columns(min(3, len(cols_adstock)))
            for i, col in enumerate(cols_adstock):
                with grid[i % 3]:
                    st.markdown(f"**{col}**")
                    prev = st.session_state.adstock_params.get(col, {})
                    decay  = st.slider(f"Decay ({col})",  0.0, 1.0, prev.get("fdecayRate", 0.5), 0.01, key=f"d_{col}")
                    peak   = st.number_input(f"Peak ({col})", 1, 52, prev.get("peak", 1), key=f"p_{col}")
                    length = st.number_input(f"Length ({col})", peak+1, 600, prev.get("length", 82), key=f"l_{col}")
                    adstock_params[col] = {"fdecayRate": decay, "peak": int(peak), "length": int(length)}

            if st.button("▶ Aplicar Adstock", type="primary"):
                df_ad = df.copy()
                errors = []
                for col, params in adstock_params.items():
                    try:
                        df_ad[col] = adstockv3_v1(df[col], **params)
                    except Exception as e:
                        errors.append(f"{col}: {e}")
                if errors:
                    st.error("\n".join(errors))
                else:
                    st.session_state.df_adstock  = df_ad
                    st.session_state.adstock_params = adstock_params
                    st.success("✅ Adstock aplicado correctamente.")

            if st.session_state.df_adstock is not None and cols_adstock:
                st.markdown("### Comparación antes / después")
                sel_vis = st.selectbox("Variable a visualizar", cols_adstock, key="vis_adstock")
                if sel_vis:
                    fig, ax = plt.subplots(figsize=(14, 4))
                    ax.plot(df[st.session_state.fecha_col], df[sel_vis], label="Original", color="#94a3b8")
                    ax.plot(df[st.session_state.fecha_col],
                            st.session_state.df_adstock[sel_vis], label="Adstock", color="#2563eb", lw=1.5)
                    ax.legend(); ax.grid(True, linestyle="--", alpha=0.3)
                    plt.xticks(rotation=45); fig.tight_layout()
                    st.pyplot(fig)

# ══════════════════════════════════════════════
#  TAB 3 – HILL
# ══════════════════════════════════════════════
with tab3:
    st.header("Transformación Hill (S-curve)")
    base_df = st.session_state.df_adstock if st.session_state.df_adstock is not None else st.session_state.df_raw
    if base_df is None:
        st.info("Carga un dataset (y opcionalmente aplica Adstock primero).")
    else:
        df = base_df.copy()
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cols_hill = st.multiselect("Variables para Hill", num_cols)

        hill_params = {}
        if cols_hill:
            st.markdown("### Parámetros Hill por variable")
            grid = st.columns(min(3, len(cols_hill)))
            for i, col in enumerate(cols_hill):
                with grid[i % 3]:
                    st.markdown(f"**{col}**")
                    prev = st.session_state.hill_params.get(col, {})
                    media_val = float(df[col].mean()) if df[col].mean() != 0 else 1.0
                    rho   = st.number_input(f"rho ({col})", value=prev.get("rho", round(media_val, 2)), format="%.2f", key=f"hr_{col}")
                    p_v   = st.slider(f"p ({col})", 0.1, 5.0, prev.get("p", 1.0), 0.1, key=f"hp_{col}")
                    #beta  = st.slider(f"beta ({col})", 0.1, 10.0, prev.get("beta", 1.0), 0.1, key=f"hb_{col}")
                    #alpha = st.number_input(f"alpha ({col})", value=prev.get("alpha", 0.0), format="%.4f", key=f"ha_{col}")
                    hill_params[col] = {"rho": rho, "p": p_v, "beta": beta, "alpha": alpha}

            if st.button("▶ Aplicar Hill", type="primary"):
                df_h = df.copy()
                for col, params in hill_params.items():
                    df_h[col] = hill(df[col], **params)
                st.session_state.df_hill    = df_h
                st.session_state.hill_params = hill_params
                st.success("✅ Hill aplicado.")

            if st.session_state.df_hill is not None and cols_hill:
                sel_vis = st.selectbox("Variable a visualizar", cols_hill, key="vis_hill")
                if sel_vis:
                    fig, ax = plt.subplots(figsize=(14, 4))
                    ax.plot(df[st.session_state.fecha_col], df[sel_vis], label="Antes Hill", color="#94a3b8")
                    ax.plot(df[st.session_state.fecha_col],
                            st.session_state.df_hill[sel_vis], label="Después Hill", color="#7c3aed", lw=1.5)
                    ax.legend(); ax.grid(True, linestyle="--", alpha=0.3)
                    plt.xticks(rotation=45); fig.tight_layout()
                    st.pyplot(fig)

# ══════════════════════════════════════════════
#  TAB 4 – REZAGOS Y DIFERENCIAS
# ══════════════════════════════════════════════
with tab4:
    st.header("Rezagos y Diferencias")
    if st.session_state.get("df_hill") is not None:
        base_df = st.session_state.get("df_hill")
    elif st.session_state.get("df_adstock") is not None:
        base_df = st.session_state.get("df_adstock")
    else:
        base_df = st.session_state.get("df_raw")

    if base_df is None:
        st.info("Carga un dataset primero.")
    else:
        df = base_df.copy()
        num_cols = df.select_dtypes(include=np.number).columns.tolist()

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Rezagos")
            cols_lag = st.multiselect("Variables a rezagar", num_cols, key="lag_vars")
            lag_params = {}
            for col in cols_lag:
                n = st.number_input(f"Periodos lag ({col})", 1, 52, 1, key=f"lag_{col}")
                lag_params[col] = int(n)

        with c2:
            st.markdown("#### Diferencias")
            cols_diff = st.multiselect("Variables a diferenciar", num_cols, key="diff_vars")
            diff_params = {}
            for col in cols_diff:
                n = st.number_input(f"Orden diferencia ({col})", 1, 4, 1, key=f"diff_{col}")
                diff_params[col] = int(n)

        # Columnas derivadas manuales
        st.markdown("#### Columnas derivadas (suma/resta)")
        with st.expander("Crear columna combinada"):
            new_col_name = st.text_input("Nombre nueva columna", "Raw_Inv")
            cols_suma = st.multiselect("Columnas a sumar", num_cols)
            if st.button("Crear columna") and cols_suma:
                df[new_col_name] = df[cols_suma].sum(axis=1)
                st.success(f"Columna '{new_col_name}' creada.")

        # Filtro de fechas
        st.markdown("#### Filtro de fechas")
        col_f = st.session_state.fecha_col
        if col_f:
            try:
                df[col_f] = pd.to_datetime(df[col_f])
                min_d = df[col_f].min().date()
                max_d = df[col_f].max().date()
                d1, d2 = st.date_input("Rango", [min_d, max_d], min_value=min_d, max_value=max_d)
            except:
                d1, d2 = None, None
        else:
            d1, d2 = None, None

        if st.button("▶ Aplicar rezagos / diferencias", type="primary"):
            df_r = df.copy()
            if col_f and d1 and d2:
                df_r = df_r[(df_r[col_f] >= pd.Timestamp(d1)) & (df_r[col_f] <= pd.Timestamp(d2))]

            for col, n in lag_params.items():
                df_r[f"{col}_lag{n}"] = df_r[col].shift(n)

            for col, n in diff_params.items():
                df_r[f"{col}_d{n}"] = df_r[col].diff(n)

            st.session_state.df_rezagos  = df_r
            st.session_state.lag_params  = lag_params
            st.session_state.diff_params = diff_params
            st.success(f"✅ Dataset preparado: {df_r.shape}")
            st.dataframe(df_r.head(20), use_container_width=True)

# ══════════════════════════════════════════════
#  TAB 5 – MODELO OLS
# ══════════════════════════════════════════════
with tab5:
    st.header("Modelado OLS – MMM")

    if st.session_state.get("df_rezagos") is not None:
        df_model_base = st.session_state.get("df_rezagos")
    elif st.session_state.get("df_hill") is not None:
        df_model_base = st.session_state.get("df_hill")
    elif st.session_state.get("df_adstock") is not None:
        df_model_base = st.session_state.get("df_adstock")
    else:
        df_model_base = st.session_state.get("df_raw")

    if df_model_base is None:
        st.info("Completa los pasos anteriores primero.")
    else:
        df_m = df_model_base.copy()

        # Asegurar variables en session_state
        if "x_cols" not in st.session_state:
            st.session_state.x_cols = []

        x_cols = st.session_state.x_cols

        # ─────────────────────────────────────────
        # AUTO MODELO BASE (si aún no existe)
        # ─────────────────────────────────────────
        if st.session_state.modelo is None and st.session_state.target_col:

            num_cols = df_m.select_dtypes(include=np.number).columns.tolist()
            auto_x = [c for c in num_cols if c != st.session_state.target_col]
            auto_x = auto_x[:8]

            if auto_x:
                try:
                    modelo_auto, contri_auto = ajustar_ols(
                        df_m,
                        st.session_state.target_col,
                        auto_x
                    )

                    st.session_state.modelo = modelo_auto
                    st.session_state.contri = contri_auto
                    st.session_state.x_cols = auto_x
                    x_cols = auto_x

                    st.success("🤖 Modelo base generado automáticamente.")

                except Exception as e:
                    st.warning(f"No se pudo generar modelo automático: {e}")

        # Botón para regenerar modelo automático
        if st.button("🔄 Regenerar Modelo Automático"):
            st.session_state.modelo = None
            st.session_state.contri = None
            st.rerun()

        # ── Selector manual de variables ─────────────────────────────
        st.markdown("### Variables del Modelo")
        num_cols = df_m.select_dtypes(include=np.number).columns.tolist()
        posibles_x = [c for c in num_cols if c != st.session_state.target_col]

        x_cols = st.multiselect(
            "Selecciona variables independientes",
            posibles_x,
            default=st.session_state.x_cols
        )

        # ── Restricciones de contribución ──────────────────────────────
        st.markdown("### Restricciones de contribución (%)")
        st.caption("Verde = dentro del rango | Rojo = fuera del rango.")

        limites = {}

        with st.expander("Configurar rangos objetivo"):

            inv_cols = st.multiselect(
                "Inversión propia (7–12%)",
                x_cols,
                key="inv_propia"
            )

            comp_cols = st.multiselect(
                "Competencia / IBOPE (5–9%)",
                x_cols,
                key="inv_comp"
            )

            otros_cols = st.multiselect(
                "Promo / Estacionalidad (<5%)",
                x_cols,
                key="otros"
            )

            for c in inv_cols:
                limites[c] = (7.0, 12.0)
            for c in comp_cols:
                limites[c] = (-9.0, -5.0)
            for c in otros_cols:
                limites[c] = (-5.0, 5.0)

        # ── Ajustar modelo manual ─────────────────────────────────────
        if st.button("▶ Ajustar Modelo OLS", type="primary", disabled=not x_cols):
            try:
                modelo, contri = ajustar_ols(
                    df_m,
                    st.session_state.target_col,
                    x_cols
                )

                st.session_state.modelo = modelo
                st.session_state.contri = contri
                st.session_state.x_cols = x_cols

            except Exception as e:
                st.error(f"Error al ajustar modelo: {e}")

        # ── RESULTADOS ────────────────────────────────────────────────
        if st.session_state.modelo is not None:

            modelo = st.session_state.modelo
            contri = st.session_state.contri

            # ── MÉTRICAS ──────────────────────────────────────────────
            metricas = calcular_metricas(modelo)

            st.markdown("### Métricas del Modelo")

            col1, col2, col3 = st.columns(3)
            col1.metric("R²", metricas.get("R²", "NA"))
            col2.metric("Adj R²", metricas.get("Adj R²", "NA"))
            col3.metric("MAPE", f'{metricas.get("MAPE", "NA")}%')
            
            if metricas["R²"] < 0.8:
                st.warning("⚠️ R² bajo. Considera agregar variables o ajustar transformaciones.")
            else:
                st.success("✅ Buen nivel explicativo del modelo.")

            # ── CONTRIBUCIONES ───────────────────────────────────────
            st.markdown("### Contribuciones por Variable")

            df_c = pd.DataFrame({
                "Variable": list(contri.keys()),
                "Contribución (%)": [round(v, 2) for v in contri.values()]
            }).sort_values("Contribución (%)", ascending=False)

            def semaforo(row):
                var = row["Variable"]
                val = row["Contribución (%)"]
                lim = limites.get(var)
                if lim:
                    return "🟢" if lim[0] <= val <= lim[1] else "🔴"
                return "⚪"

            df_c["Estado"] = df_c.apply(semaforo, axis=1)

            st.dataframe(df_c, use_container_width=True, hide_index=True)

            fig_c = plot_contrib(contri, limites)
            st.pyplot(fig_c)

            # ── SUMMARY OLS ──────────────────────────────────────────
            with st.expander("📄 Summary OLS completo"):
                st.text(modelo.summary().as_text())

            # ── COEFICIENTES ─────────────────────────────────────────
            st.markdown("### Coeficientes")

            df_coef = pd.DataFrame({
                "Variable": modelo.params.index,
                "Coeficiente": modelo.params.values,
                "p-value": modelo.pvalues.values,
            })

            st.dataframe(
                df_coef.style.format({
                    "Coeficiente": "{:.4f}",
                    "p-value": "{:.4f}"
                }),
                use_container_width=True,
                hide_index=True
            )

            # ── FIT ──────────────────────────────────────────────────
            st.markdown("### Ajuste del Modelo")

            try:
                fig_fit = plot_fit(
                    df_m,
                    modelo,
                    st.session_state.target_col,
                    st.session_state.fecha_col
                )
                st.pyplot(fig_fit)
            except Exception as e:
                st.warning(f"No se pudo graficar el fit: {e}")

            # ── DIAGNÓSTICOS ─────────────────────────────────────────
            st.markdown("### Diagnósticos de Residuales")

            fig2, axes = plt.subplots(1, 3, figsize=(15, 4))
            resid = modelo.resid

            axes[0].hist(resid, bins=25)
            axes[0].set_title("Distribución Residuales")

            sm.qqplot(resid, line="s", ax=axes[1])
            axes[1].set_title("Q-Q Plot")

            axes[2].scatter(modelo.fittedvalues, resid)
            axes[2].axhline(0)
            axes[2].set_title("Residuales vs Ajustados")

            fig2.tight_layout()
            st.pyplot(fig2)

            # ── EXPORTAR ─────────────────────────────────────────────
            st.markdown("### Exportar")

            st.download_button(
                "⬇️ Descargar contribuciones (.csv)",
                df_c.to_csv(index=False).encode("utf-8"),
                "contribuciones_mmm.csv",
                "text/csv"
            )

            st.download_button(
                "⬇️ Descargar coeficientes (.csv)",
                df_coef.to_csv(index=False).encode("utf-8"),
                "coeficientes_mmm.csv",
                "text/csv"
            )
