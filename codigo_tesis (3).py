"""
================================================================================
Codigo de la tesis de Licenciatura en Finanzas (Universidad de San Andres)

    "La reaccion del mercado frente a los anuncios de dividendos:
     evidencia del mercado argentino mediante un Event Study"

Autores : Ezequiel Mario Harari y Maria Candelaria Seleme Duran
Ano     : 2026

Este script reproduce el procesamiento de datos, el calculo de los retornos
anormales (AR) y acumulados (CAR), las pruebas de significatividad parametricas
(test t) y no parametricas (Wilcoxon, sign test), los procedimientos bootstrap y
la totalidad de los modelos de regresion reportados en el trabajo.

Requisitos: Python 3.10+, pandas, numpy, scipy, statsmodels, openpyxl.
Los datos de entrada (Bloomberg, serie PX_LAST) no se distribuyen en este
repositorio por restricciones de licencia.
================================================================================
"""

import pandas as pd
from scipy.stats import ttest_1samp, wilcoxon, binomtest
import numpy as np


# ============================================================
# 1. CARGAR BASE
# ============================================================

archivo = "/Users/Usuario/Documents/Tesis/Dividendos_datos_new.xlsx"
df = pd.read_excel(archivo, sheet_name="Dividendos_val")


# ============================================================
# 2. CREAR AR
# ============================================================

def crear_AR(df, evento="anuncio", ventana=2):
    df = df.copy()

    columnas = {
        "anuncio": {
            -2: ("Precio-1 dia", "Precio-2 dia", "Precio-1 dia.1", "Precio-2 dia.1"),
            -1: ("Retorno -1D", "Retorno -1D.1"),
             0: ("Retorno fecha de anuncio", "Retorno fecha de anuncioM"),
             1: ("Retorno +1_D", "Retorno +1_DM"),
             2: ("Precio+2_dia", "Precio+1_dia", "Precio+2_dia.1", "Precio+1_dia.1"),
        },
        "pago": {
            -2: ("Precio_-1_dia_p_date", "Precio_-2_dia_p_date", "Precio_-1_dia_p_date.1", "Precio_-2_dia_p_date.1"),
            -1: ("Retorno pmt_-1D", "Retorno pmt_-1D.1"),
             0: ("Retorno pmt_D", "Retorno pmt_D.1"),
             1: ("Retonro pmt_+1D", "Retonro pmt_+1D.1"),
             2: ("Precio_+2_dia_p_date", "Precio_+1_dia_p_date", "Precio_+2_dia_p_date.1", "Precio_+1_dia_p_date.1"),
        }
    }

    for dia in range(-ventana, ventana + 1):
        datos = columnas[evento][dia]
        nombre_ar = f"AR_{evento}_{dia:+d}"

        if dia in [-2, 2]:
            precio_accion_hoy, precio_accion_ayer, precio_mkt_hoy, precio_mkt_ayer = datos

            retorno_accion = df[precio_accion_hoy] / df[precio_accion_ayer] - 1
            retorno_mercado = df[precio_mkt_hoy] / df[precio_mkt_ayer] - 1

            df[nombre_ar] = retorno_accion - retorno_mercado

        else:
            retorno_accion, retorno_mercado = datos
            df[nombre_ar] = df[retorno_accion] - df[retorno_mercado]

    return df


# ============================================================
# 3. CREAR CAR
# ============================================================

def crear_CAR(df, evento="anuncio", ventana=1):
    df = df.copy()

    columnas_ar = [
        f"AR_{evento}_{dia:+d}"
        for dia in range(-ventana, ventana + 1)
    ]

    nombre_car = f"CAR_{evento}_[{-ventana},+{ventana}]"

    df[nombre_car] = df[columnas_ar].sum(axis=1, skipna=False)

    return df


# ============================================================
# 4. VARIABLES DE RESULTADO
# ============================================================

def obtener_variables_resultado():
    return [
        "AR_anuncio_+0",
        "AR_pago_+0",
        "CAR_anuncio_[-1,+1]",
        "CAR_pago_[-1,+1]",
        "CAR_anuncio_[-2,+2]",
        "CAR_pago_[-2,+2]"
    ]


# ============================================================
# 5. RESUMEN DESCRIPTIVO GENERAL
# ============================================================

def crear_resumen_descriptivo(df):
    variables = obtener_variables_resultado()

    resumen = df[variables].describe().T

    resumen["mean_%"] = resumen["mean"] * 100
    resumen["std_%"] = resumen["std"] * 100

    return resumen


# ============================================================
# 6. TESTS DE SIGNIFICATIVIDAD GENERAL (t-test simple, original)
# ============================================================

def crear_tests_significatividad(df):
    variables = obtener_variables_resultado()

    resultados = []

    for var in variables:
        serie = df[var].dropna()

        if len(serie) > 1:
            t_stat, p_value = ttest_1samp(serie, 0)
        else:
            t_stat, p_value = np.nan, np.nan

        resultados.append({
            "Variable": var,
            "Media": serie.mean(),
            "Media_%": serie.mean() * 100,
            "T-stat": t_stat,
            "P-value": p_value,
            "N": len(serie)
        })

    return pd.DataFrame(resultados)


# ============================================================
# 6.bis  BATERIA DE TESTS DE SIGNIFICATIVIDAD (H0: media/mediana = 0)
# ------------------------------------------------------------
# Para una serie de AR o CAR aplica cuatro contrastes:
#   1. t-test cross-sectional   -> asume normalidad de la media
#   2. Wilcoxon signed-rank     -> no parametrico, contrasta la mediana
#   3. Sign test (binomial)     -> proporcion de AR positivos vs 50%
#   4. Bootstrap                -> p-valor empirico + IC95 sin asumir normalidad
# Los tres ultimos son robustos a colas pesadas y outliers, habituales
# en retornos. Util para mostrar si el efecto es general o lo arrastran
# unos pocos eventos extremos.
# ============================================================

def _bootstrap_media(serie, n_boot=10000, alpha=0.05, seed=42):
    """p-valor bilateral bajo H0: media=0 e IC percentil al (1-alpha)."""
    x = serie.to_numpy()
    n = len(x)
    if n < 2:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = x[idx].mean(axis=1)

    obs = x.mean()
    centradas = boot_means - obs                       # mundo donde H0 (media=0) es cierta
    p_valor = np.mean(np.abs(centradas) >= np.abs(obs))

    ic_low = np.percentile(boot_means, 100 * alpha / 2)
    ic_high = np.percentile(boot_means, 100 * (1 - alpha / 2))

    return p_valor, ic_low, ic_high


def _tests_una_serie(serie):
    """Aplica los cuatro tests a una serie ya limpia (sin NaN)."""
    n = len(serie)

    fila = {
        "N": n,
        "Media_%": serie.mean() * 100 if n else np.nan,
        "Mediana_%": serie.median() * 100 if n else np.nan,
    }

    if n <= 1:
        fila.update({
            "%_positivos": np.nan,
            "t_stat": np.nan, "p_t": np.nan,
            "Wilcoxon_stat": np.nan, "p_wilcoxon": np.nan,
            "p_sign": np.nan,
            "p_bootstrap": np.nan, "IC95_low_%": np.nan, "IC95_high_%": np.nan,
        })
        return fila

    # 1. t-test cross-sectional
    t_stat, p_t = ttest_1samp(serie, 0)

    # 2. Wilcoxon signed-rank (descarta ceros exactos)
    sin_ceros = serie[serie != 0]
    try:
        w_stat, p_w = wilcoxon(sin_ceros)
    except ValueError:
        w_stat, p_w = np.nan, np.nan

    # 3. Sign test: proporcion de positivos contra 0.5
    positivos = int((serie > 0).sum())
    no_nulos = int((serie != 0).sum())
    p_sign = binomtest(positivos, no_nulos, 0.5).pvalue if no_nulos else np.nan
    pct_pos = positivos / no_nulos * 100 if no_nulos else np.nan

    # 4. Bootstrap
    p_boot, ic_low, ic_high = _bootstrap_media(serie)

    fila.update({
        "%_positivos": pct_pos,
        "t_stat": t_stat, "p_t": p_t,
        "Wilcoxon_stat": w_stat, "p_wilcoxon": p_w,
        "p_sign": p_sign,
        "p_bootstrap": p_boot,
        "IC95_low_%": ic_low * 100,
        "IC95_high_%": ic_high * 100,
    })
    return fila


def crear_tests_robustos(df):
    """Bateria completa de tests para todas las variables de resultado."""
    variables = obtener_variables_resultado()

    resultados = []
    for var in variables:
        serie = df[var].dropna()
        fila = {"Variable": var}
        fila.update(_tests_una_serie(serie))
        resultados.append(fila)

    return pd.DataFrame(resultados)


def crear_tests_robustos_por_grupo(df, grupo):
    """Bateria completa de tests por cada nivel de la variable de grupo."""
    variables = obtener_variables_resultado()

    resultados = []
    for nombre_grupo, data in df.groupby(grupo, observed=False):
        for var in variables:
            serie = data[var].dropna()
            fila = {"Grupo": nombre_grupo, "Variable": var}
            fila.update(_tests_una_serie(serie))
            resultados.append(fila)

    return pd.DataFrame(resultados)


# ============================================================
# 7. CREAR TIPO DE DIVIDENDO AGRUPADO
# ============================================================

def crear_tipo_dividendo_agrupado(df):
    df = df.copy()

    def clasificar_tipo(tipo):
        if tipo in ["Regular Cash", "Final", "Interim"]:
            return "Cash ordinario"
        elif tipo in ["Special Cash", "Return of Capital"]:
            return "Cash extraordinario"
        elif tipo in ["Stock Dividend", "Stock Split"]:
            return "Acciones / split"
        elif tipo in ["Rights Issue", "Entitlement"]:
            return "Derechos / entitlement"
        elif tipo == "Spinoff":
            return "Spinoff"
        elif tipo == "Cancelled":
            return "Cancelado"
        else:
            return "Otros"

    df["Tipo_Dividendo_Agrupado"] = df["Type"].apply(clasificar_tipo)

    return df


# ============================================================
# 8. CREAR GRUPOS DE DIVIDEND YIELD
# ============================================================

def crear_grupos_dividend_yield(
    df,
    columna_yield="Dividend Yield anuncio",
    q=3,
    nombre_columna=None
):

    df = df.copy()

    if nombre_columna is None:
        nombre_columna = f"Grupo_Dividend_Yield_Q{q}"

    labels = [f"Q{i}" for i in range(1, q + 1)]

    df[nombre_columna] = pd.qcut(
        df[columna_yield],
        q=q,
        labels=labels,
        duplicates="drop"
    )

    return df
# ============================================================
# 9. RESUMEN DESCRIPTIVO POR GRUPO
# ============================================================

def crear_resumen_por_grupo(df, grupo):
    variables = obtener_variables_resultado()

    resumen = (
        df
        .groupby(grupo, observed=False)[variables]
        .agg(["count", "mean", "std", "min", "median", "max"])
    )

    return resumen


# ============================================================
# 10. TESTS DE SIGNIFICATIVIDAD POR GRUPO (t-test simple, original)
# ============================================================

def crear_tests_por_grupo(df, grupo):
    variables = obtener_variables_resultado()

    resultados = []

    for nombre_grupo, data in df.groupby(grupo, observed=False):
        for var in variables:
            serie = data[var].dropna()

            if len(serie) > 1:
                t_stat, p_value = ttest_1samp(serie, 0)
            else:
                t_stat, p_value = np.nan, np.nan

            resultados.append({
                "Grupo": nombre_grupo,
                "Variable": var,
                "N": len(serie),
                "Media": serie.mean(),
                "Media_%": serie.mean() * 100,
                "T-stat": t_stat,
                "P-value": p_value
            })

    return pd.DataFrame(resultados)


# ============================================================
# 11. EJECUTAR TODO
# ============================================================

df_final = df.copy()

# Crear AR
df_final = crear_AR(df_final, evento="anuncio", ventana=2)
df_final = crear_AR(df_final, evento="pago", ventana=2)

# Crear CAR [-1,+1]
df_final = crear_CAR(df_final, evento="anuncio", ventana=1)
df_final = crear_CAR(df_final, evento="pago", ventana=1)

# Crear CAR [-2,+2]
df_final = crear_CAR(df_final, evento="anuncio", ventana=2)
df_final = crear_CAR(df_final, evento="pago", ventana=2)

# Crear segmentaciones
df_final = crear_tipo_dividendo_agrupado(df_final)

df_final = crear_grupos_dividend_yield(
    df_final,
    columna_yield="Dividend Yield anuncio",
    q=3,
    nombre_columna="Grupo_Dividend_Yield_Q3"
)

# Cuartiles
df_final = crear_grupos_dividend_yield(
    df_final,
    columna_yield="Dividend Yield anuncio",
    q=4,
    nombre_columna="Grupo_Dividend_Yield_Q4"
)

# Quintiles
df_final = crear_grupos_dividend_yield(
    df_final,
    columna_yield="Dividend Yield anuncio",
    q=5,
    nombre_columna="Grupo_Dividend_Yield_Q5"
)
# Resultados generales
resumen_general = crear_resumen_descriptivo(df_final)
tests_generales = crear_tests_significatividad(df_final)
tests_robustos_generales = crear_tests_robustos(df_final)

# Resultados por tipo de dividendo agrupado
resumen_tipo = crear_resumen_por_grupo(
    df_final,
    grupo="Tipo_Dividendo_Agrupado"
)

tests_tipo = crear_tests_por_grupo(
    df_final,
    grupo="Tipo_Dividendo_Agrupado"
)

tests_robustos_tipo = crear_tests_robustos_por_grupo(
    df_final,
    grupo="Tipo_Dividendo_Agrupado"
)

# Resultados por Dividend Yield
# ======================
# Yield Q3
# ======================

resumen_yield_q3 = crear_resumen_por_grupo(
    df_final,
    grupo="Grupo_Dividend_Yield_Q3"
)

tests_yield_q3 = crear_tests_por_grupo(
    df_final,
    grupo="Grupo_Dividend_Yield_Q3"
)

tests_robustos_yield_q3 = crear_tests_robustos_por_grupo(
    df_final,
    grupo="Grupo_Dividend_Yield_Q3"
)

# ======================
# Yield Q4
# ======================

resumen_yield_q4 = crear_resumen_por_grupo(
    df_final,
    grupo="Grupo_Dividend_Yield_Q4"
)

tests_yield_q4 = crear_tests_por_grupo(
    df_final,
    grupo="Grupo_Dividend_Yield_Q4"
)

tests_robustos_yield_q4 = crear_tests_robustos_por_grupo(
    df_final,
    grupo="Grupo_Dividend_Yield_Q4"
)

# ======================
# Yield Q5
# ======================

resumen_yield_q5 = crear_resumen_por_grupo(
    df_final,
    grupo="Grupo_Dividend_Yield_Q5"
)

tests_yield_q5 = crear_tests_por_grupo(
    df_final,
    grupo="Grupo_Dividend_Yield_Q5"
)

tests_robustos_yield_q5 = crear_tests_robustos_por_grupo(
    df_final,
    grupo="Grupo_Dividend_Yield_Q5"
)


# ============================================================
# 12. EXPORTAR EXCEL
# ============================================================

salida = "/Users/Usuario/Documents/Tesis/resultados_tesis.xlsx"

with pd.ExcelWriter(salida, engine="openpyxl") as writer:
    df_final.to_excel(writer, sheet_name="Base_AR_CAR", index=False)
    resumen_general.to_excel(writer, sheet_name="Resumen_general")
    tests_generales.to_excel(writer, sheet_name="Tests_generales", index=False)
    tests_robustos_generales.to_excel(writer, sheet_name="Tests_robustos_gen", index=False)
    resumen_tipo.to_excel(writer, sheet_name="Resumen_tipo")
    tests_tipo.to_excel(writer, sheet_name="Tests_tipo", index=False)
    tests_robustos_tipo.to_excel(writer, sheet_name="Tests_robustos_tipo", index=False)

    # Yield Q3
    resumen_yield_q3.to_excel(writer, sheet_name="Resumen_yield_Q3")
    tests_yield_q3.to_excel(writer, sheet_name="Tests_yield_Q3", index=False)
    tests_robustos_yield_q3.to_excel(writer, sheet_name="Tests_robustos_Q3", index=False)

    # Yield Q4
    resumen_yield_q4.to_excel(writer, sheet_name="Resumen_yield_Q4")
    tests_yield_q4.to_excel(writer, sheet_name="Tests_yield_Q4", index=False)
    tests_robustos_yield_q4.to_excel(writer, sheet_name="Tests_robustos_Q4", index=False)

    # Yield Q5
    resumen_yield_q5.to_excel(writer, sheet_name="Resumen_yield_Q5")
    tests_yield_q5.to_excel(writer, sheet_name="Tests_yield_Q5", index=False)
    tests_robustos_yield_q5.to_excel(writer, sheet_name="Tests_robustos_Q5", index=False)

print(f"Excel creado en: {salida}")