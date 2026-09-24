"""
================================================================================
Codigo reproducible de la tesis de Licenciatura en Finanzas (UdeSA)

    "La reaccion del mercado frente a los anuncios de dividendos:
     evidencia del mercado argentino mediante un Event Study"

Autores : Ezequiel Mario Harari y Maria Candelaria Seleme Duran
Ano     : 2026

ESTRUCTURA DE MUESTRAS
----------------------
1) Muestra total: 774 eventos.
   Se conserva para descripcion general y comparacion por tipo de distribucion.

2) Muestra principal: 633 eventos "Regular Cash".
   Se utiliza para H1 principal, tests robustos, Dividend Yield, terciles,
   cuartiles, quintiles y TODAS las regresiones.

Este script reproduce:
- AR diarios y CAR para las ventanas usadas en el PTG/presentacion.
- Estadistica descriptiva y test t.
- Wilcoxon, Sign Test y Bootstrap.
- Analisis por tipo de dividendo.
- Analisis por terciles, cuartiles y quintiles de Dividend Yield.
- Regresiones por cuartiles: MCO clasico, HC3 y HC3 + FE por ano.
- Regresiones con Dividend Yield continuo: HC3 y HC3 + FE por ano.
- Robustez con ln(1 + Dividend Yield).
- Robustez excluyendo p1/p99 del CAR.
- Robustez winsorizando Dividend Yield al 1% en ambas colas.
- Tablas auxiliares para AAR/CAAR, binscatter, composicion, anos y sectores.

IMPORTANTE SOBRE LOS EFECTOS FIJOS POR ANO
-------------------------------------------
El PTG publicado reporta coeficientes FE que se reproducen usando la conversion
original de "Fecha de aviso" con pd.to_datetime. Como la base mezcla fechas de
Excel seriales con datetimes, esa conversion agrupa parte de los eventos antiguos
como ano 1970. Para que este script reproduzca EXACTAMENTE las tablas del PTG,
REPRODUCIR_FE_PTG=True por defecto.

El script tambien calcula el ano calendario correctamente (incluyendo 1996-2016)
y lo exporta como diagnostico. Si en algun momento se decide reestimar los FE con
los 30 anos correctamente parseados, cambiar REPRODUCIR_FE_PTG=False. Eso cambia
los coeficientes FE y, por lo tanto, ya no coincide exactamente con el PTG.

Requisitos:
Python 3.10+, pandas, numpy, scipy, statsmodels, openpyxl.
================================================================================
"""

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, wilcoxon, binomtest
from scipy.stats.mstats import winsorize
import statsmodels.formula.api as smf


# ============================================================
# 0. CONFIGURACION
# ============================================================

archivo = "/Users/Usuario/Documents/Tesis/Dividendos_datos_new.xlsx"
hoja_base = "Dividendos_val"

salida = "/Users/Usuario/Documents/Tesis/resultados_tesis_completo.xlsx"

# True = reproduce exactamente los FE reportados en el PTG.
# False = usa el ano calendario correctamente parseado para 1996-2026.
REPRODUCIR_FE_PTG = False

N_BOOT = 10000
SEED_BOOT = 42


# ============================================================
# 1. CARGAR BASE
# ============================================================

archivo_path = Path(archivo)
if not archivo_path.exists():
    raise FileNotFoundError(
        f"No se encontro el archivo de entrada:\n{archivo}\n\n"
        "Modificar la variable 'archivo' al comienzo del script."
    )

df = pd.read_excel(archivo, sheet_name=hoja_base)


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
        },
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
# 3. CREAR CAR PARA CUALQUIER RANGO
# ============================================================

def _fmt_dia(dia):
    return "0" if dia == 0 else f"{dia:+d}"


def crear_CAR_rango(df, evento="anuncio", inicio=-1, fin=1):
    df = df.copy()

    columnas_ar = [
        f"AR_{evento}_{dia:+d}"
        for dia in range(inicio, fin + 1)
    ]

    nombre_car = f"CAR_{evento}_[{_fmt_dia(inicio)},{_fmt_dia(fin)}]"
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
        "CAR_pago_[-2,+2]",
        "CAR_anuncio_[0,+1]",
        "CAR_pago_[0,+1]",
        "CAR_anuncio_[0,+2]",
        "CAR_pago_[0,+2]",
    ]


# ============================================================
# 5. RESUMEN DESCRIPTIVO
# ============================================================

def crear_resumen_descriptivo(df):
    variables = obtener_variables_resultado()
    resumen = df[variables].describe().T
    resumen["mean_%"] = resumen["mean"] * 100
    resumen["std_%"] = resumen["std"] * 100
    resumen["median_%"] = df[variables].median() * 100
    return resumen


# ============================================================
# 6. TEST T
# ============================================================

def crear_tests_significatividad(df):
    resultados = []

    for var in obtener_variables_resultado():
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
            "N": len(serie),
        })

    return pd.DataFrame(resultados)


# ============================================================
# 7. TESTS ROBUSTOS: T, WILCOXON, SIGN Y BOOTSTRAP
# ============================================================

def _bootstrap_media(serie, n_boot=N_BOOT, alpha=0.05, seed=SEED_BOOT):
    """Bootstrap de la media: p-valor bilateral bajo H0: media=0 e IC percentil."""
    x = serie.to_numpy(dtype=float)
    n = len(x)

    if n < 2:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = x[idx].mean(axis=1)

    obs = x.mean()
    centradas = boot_means - obs
    p_valor = np.mean(np.abs(centradas) >= np.abs(obs))

    ic_low = np.percentile(boot_means, 100 * alpha / 2)
    ic_high = np.percentile(boot_means, 100 * (1 - alpha / 2))

    return p_valor, ic_low, ic_high


def _tests_una_serie(serie):
    serie = serie.dropna()
    n = len(serie)

    fila = {
        "N": n,
        "Media_%": serie.mean() * 100 if n else np.nan,
        "Mediana_%": serie.median() * 100 if n else np.nan,
    }

    if n <= 1:
        fila.update({
            "%_positivos": np.nan,
            "t_stat": np.nan,
            "p_t": np.nan,
            "Wilcoxon_stat": np.nan,
            "p_wilcoxon": np.nan,
            "p_sign": np.nan,
            "p_bootstrap": np.nan,
            "IC95_low_%": np.nan,
            "IC95_high_%": np.nan,
        })
        return fila

    # 1. t-test cross-sectional
    t_stat, p_t = ttest_1samp(serie, 0)

    # 2. Wilcoxon signed-rank: descarta ceros exactos
    sin_ceros = serie[serie != 0]
    try:
        w_stat, p_w = wilcoxon(sin_ceros)
    except ValueError:
        w_stat, p_w = np.nan, np.nan

    # 3. Sign test: positivos vs 50%
    positivos = int((serie > 0).sum())
    no_nulos = int((serie != 0).sum())
    p_sign = binomtest(positivos, no_nulos, 0.5).pvalue if no_nulos else np.nan
    pct_pos = positivos / no_nulos * 100 if no_nulos else np.nan

    # 4. Bootstrap
    p_boot, ic_low, ic_high = _bootstrap_media(serie)

    fila.update({
        "%_positivos": pct_pos,
        "t_stat": t_stat,
        "p_t": p_t,
        "Wilcoxon_stat": w_stat,
        "p_wilcoxon": p_w,
        "p_sign": p_sign,
        "p_bootstrap": p_boot,
        "IC95_low_%": ic_low * 100,
        "IC95_high_%": ic_high * 100,
    })

    return fila


def crear_tests_robustos(df):
    resultados = []

    for var in obtener_variables_resultado():
        fila = {"Variable": var}
        fila.update(_tests_una_serie(df[var]))
        resultados.append(fila)

    return pd.DataFrame(resultados)


def crear_tests_robustos_por_grupo(df, grupo):
    resultados = []

    for nombre_grupo, data in df.groupby(grupo, observed=False):
        for var in obtener_variables_resultado():
            fila = {"Grupo": nombre_grupo, "Variable": var}
            fila.update(_tests_una_serie(data[var]))
            resultados.append(fila)

    return pd.DataFrame(resultados)


# ============================================================
# 8. TIPO DE DIVIDENDO
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
# 9. GRUPOS DE DIVIDEND YIELD
# ============================================================

def crear_grupos_dividend_yield(
    df,
    columna_yield="Dividend Yield anuncio",
    q=4,
    nombre_columna=None,
):
    df = df.copy()

    if nombre_columna is None:
        nombre_columna = f"Grupo_Dividend_Yield_Q{q}"

    labels = [f"Q{i}" for i in range(1, q + 1)]

    df[nombre_columna] = pd.qcut(
        df[columna_yield],
        q=q,
        labels=labels,
        duplicates="drop",
    )

    return df


# ============================================================
# 10. RESUMEN Y TESTS POR GRUPO
# ============================================================

def crear_resumen_por_grupo(df, grupo):
    return (
        df.groupby(grupo, observed=False)[obtener_variables_resultado()]
          .agg(["count", "mean", "std", "min", "median", "max"])
    )


def crear_tests_por_grupo(df, grupo):
    resultados = []

    for nombre_grupo, data in df.groupby(grupo, observed=False):
        for var in obtener_variables_resultado():
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
                "P-value": p_value,
            })

    return pd.DataFrame(resultados)


# ============================================================
# 11. FECHAS Y EFECTOS FIJOS POR ANO
# ============================================================

def convertir_fecha_excel_mixta(valor):
    """Convierte correctamente datetimes, strings y seriales de fecha de Excel."""
    if pd.isna(valor):
        return pd.NaT

    if isinstance(valor, (pd.Timestamp, datetime)):
        return pd.Timestamp(valor)

    if isinstance(valor, (int, float, np.integer, np.floating)):
        # Excel usa 1899-12-30 como origen efectivo para sus seriales.
        return pd.Timestamp("1899-12-30") + pd.to_timedelta(float(valor), unit="D")

    return pd.to_datetime(valor, errors="coerce")


def agregar_variables_regresion(df):
    df = df.copy()

    df["DY"] = pd.to_numeric(df["Dividend Yield anuncio"], errors="coerce")
    df["CAR_0_1"] = pd.to_numeric(df["CAR_anuncio_[0,+1]"], errors="coerce")
    df["CAR_0_2"] = pd.to_numeric(df["CAR_anuncio_[0,+2]"], errors="coerce")

    # Ano usado para reproducir exactamente los resultados FE del PTG.
    df["Anio_FE_PTG"] = pd.to_datetime(
        df["Fecha de aviso"], errors="coerce"
    ).dt.year

    # Ano calendario correctamente parseado, incluido como chequeo.
    df["Fecha_aviso_corregida"] = df["Fecha de aviso"].map(convertir_fecha_excel_mixta)
    df["Anio_calendario"] = df["Fecha_aviso_corregida"].dt.year

    df["Anio_FE"] = (
        df["Anio_FE_PTG"]
        if REPRODUCIR_FE_PTG
        else df["Anio_calendario"]
    )

    return df


# ============================================================
# 12. TABLAS AUXILIARES PARA PRESENTACION
# ============================================================

def crear_aar_diario(df, evento):
    resultados = []

    for dia in [-2, -1, 0, 1, 2]:
        var = f"AR_{evento}_{dia:+d}"
        serie = df[var].dropna()
        t_stat, p_val = ttest_1samp(serie, 0) if len(serie) > 1 else (np.nan, np.nan)

        resultados.append({
            "Evento": evento,
            "Dia": dia,
            "N": len(serie),
            "AAR": serie.mean(),
            "AAR_%": serie.mean() * 100,
            "T-stat": t_stat,
            "P-value": p_val,
        })

    return pd.DataFrame(resultados)


def crear_caar_desde_menos1(df, evento):
    """CAAR acumulado desde t=-1 para t=-1,0,+1,+2."""
    resultados = []

    for fin in [-1, 0, 1, 2]:
        columnas = [f"AR_{evento}_{d:+d}" for d in range(-1, fin + 1)]
        serie = df[columnas].sum(axis=1, skipna=False).dropna()
        t_stat, p_val = ttest_1samp(serie, 0) if len(serie) > 1 else (np.nan, np.nan)

        resultados.append({
            "Evento": evento,
            "Hasta_dia": fin,
            "N": len(serie),
            "CAAR": serie.mean(),
            "CAAR_%": serie.mean() * 100,
            "T-stat": t_stat,
            "P-value": p_val,
        })

    return pd.DataFrame(resultados)


def crear_binscatter(df, n_bins=20):
    aux = df[["DY", "CAR_0_1"]].dropna().copy()
    aux["Bin"] = pd.qcut(aux["DY"], q=n_bins, labels=False, duplicates="drop") + 1

    return (
        aux.groupby("Bin", observed=False)
           .agg(
               N=("DY", "size"),
               DY_promedio=("DY", "mean"),
               CAR_0_1_promedio=("CAR_0_1", "mean"),
           )
           .reset_index()
    )


# ============================================================
# 13. FUNCIONES PARA REGRESIONES
# ============================================================

def ajustar_ols(formula, data, hc3=False):
    """MCO clasico o MCO con matriz robusta HC3. HC3 usa inferencia z."""
    if hc3:
        return smf.ols(formula, data=data).fit(cov_type="HC3", use_t=False)
    return smf.ols(formula, data=data).fit()


def extraer_modelo(modelo, nombre_modelo, ventana, variable_clave=None):
    """Devuelve tabla completa de coeficientes y resumen del modelo."""
    filas = []

    for variable in modelo.params.index:
        filas.append({
            "Modelo": nombre_modelo,
            "Ventana": ventana,
            "Variable": variable,
            "Coeficiente": modelo.params[variable],
            "Error_estandar": modelo.bse[variable],
            "Estadistico": modelo.tvalues[variable],
            "P_value": modelo.pvalues[variable],
            "R2": modelo.rsquared,
            "R2_ajustado": modelo.rsquared_adj,
            "F": float(modelo.fvalue) if modelo.fvalue is not None else np.nan,
            "Prob_F": float(modelo.f_pvalue) if modelo.f_pvalue is not None else np.nan,
            "N": int(modelo.nobs),
            "Variable_clave": variable_clave,
        })

    return pd.DataFrame(filas)


def fila_clave(modelo, nombre_modelo, ventana, variable):
    return {
        "Modelo": nombre_modelo,
        "Ventana": ventana,
        "Variable": variable,
        "Coeficiente": modelo.params[variable],
        "Error_estandar": modelo.bse[variable],
        "P_value": modelo.pvalues[variable],
        "R2": modelo.rsquared,
        "R2_ajustado": modelo.rsquared_adj,
        "F": float(modelo.fvalue) if modelo.fvalue is not None else np.nan,
        "Prob_F": float(modelo.f_pvalue) if modelo.f_pvalue is not None else np.nan,
        "N": int(modelo.nobs),
    }


def correr_regresiones(df_regular):
    """Corre todas las especificaciones reportadas en el PTG."""
    data = df_regular.copy()

    # Asegurar Q4 con Q1 como referencia.
    data["DY_Q4"] = pd.Categorical(
        data["Grupo_Dividend_Yield_Q4"],
        categories=["Q1", "Q2", "Q3", "Q4"],
        ordered=True,
    )

    resultados_full = []
    resultados_clave = []

    # --------------------------------------------------------
    # A. REGRESIONES POR CUARTILES
    # --------------------------------------------------------
    for y, ventana in [("CAR_0_1", "[0,+1]"), ("CAR_0_2", "[0,+2]")]:
        formula_q = f'{y} ~ C(DY_Q4, Treatment(reference="Q1"))'
        formula_q_fe = f'{y} ~ C(DY_Q4, Treatment(reference="Q1")) + C(Anio_FE)'

        m_clasico = ajustar_ols(formula_q, data, hc3=False)
        m_hc3 = ajustar_ols(formula_q, data, hc3=True)
        m_fe = ajustar_ols(formula_q_fe, data, hc3=True)

        nombre_q4 = 'C(DY_Q4, Treatment(reference="Q1"))[T.Q4]'

        resultados_full.extend([
            extraer_modelo(m_clasico, "Cuartiles_MCO_clasico", ventana, nombre_q4),
            extraer_modelo(m_hc3, "Cuartiles_HC3", ventana, nombre_q4),
            extraer_modelo(m_fe, "Cuartiles_HC3_FE", ventana, nombre_q4),
        ])

        resultados_clave.extend([
            fila_clave(m_clasico, "Cuartiles_MCO_clasico", ventana, nombre_q4),
            fila_clave(m_hc3, "Cuartiles_HC3", ventana, nombre_q4),
            fila_clave(m_fe, "Cuartiles_HC3_FE", ventana, nombre_q4),
        ])

    # --------------------------------------------------------
    # B. DIVIDEND YIELD CONTINUO
    # --------------------------------------------------------
    for y, ventana in [("CAR_0_1", "[0,+1]"), ("CAR_0_2", "[0,+2]")]:
        m_cont = ajustar_ols(f"{y} ~ DY", data, hc3=True)
        m_cont_fe = ajustar_ols(f"{y} ~ DY + C(Anio_FE)", data, hc3=True)

        resultados_full.extend([
            extraer_modelo(m_cont, "DY_continuo_HC3", ventana, "DY"),
            extraer_modelo(m_cont_fe, "DY_continuo_HC3_FE", ventana, "DY"),
        ])

        resultados_clave.extend([
            fila_clave(m_cont, "DY_continuo_HC3", ventana, "DY"),
            fila_clave(m_cont_fe, "DY_continuo_HC3_FE", ventana, "DY"),
        ])

    # --------------------------------------------------------
    # C. TRANSFORMACION LOGARITMICA ln(1+DY), HC3
    #    (como Tabla 13/28 del PTG: sin FE)
    # --------------------------------------------------------
    data["ln_1p_DY"] = np.log1p(data["DY"])

    for y, ventana in [("CAR_0_1", "[0,+1]"), ("CAR_0_2", "[0,+2]")]:
        m_log = ajustar_ols(f"{y} ~ ln_1p_DY", data, hc3=True)

        resultados_full.append(
            extraer_modelo(m_log, "Log_DY_HC3", ventana, "ln_1p_DY")
        )
        resultados_clave.append(
            fila_clave(m_log, "Log_DY_HC3", ventana, "ln_1p_DY")
        )

    # --------------------------------------------------------
    # D. ROBUSTEZ: EXCLUIR OUTLIERS p1/p99 DEL CAR
    #    Cada ventana se filtra por sus propios percentiles.
    # --------------------------------------------------------
    info_outliers = []

    for y, ventana in [("CAR_0_1", "[0,+1]"), ("CAR_0_2", "[0,+2]")]:
        p1 = data[y].quantile(0.01)
        p99 = data[y].quantile(0.99)

        data_out = data.loc[(data[y] >= p1) & (data[y] <= p99)].copy()
        m_out = ajustar_ols(f"{y} ~ DY + C(Anio_FE)", data_out, hc3=True)

        resultados_full.append(
            extraer_modelo(m_out, "Sin_outliers_CAR_HC3_FE", ventana, "DY")
        )
        resultados_clave.append(
            fila_clave(m_out, "Sin_outliers_CAR_HC3_FE", ventana, "DY")
        )

        info_outliers.append({
            "Ventana": ventana,
            "P1_CAR": p1,
            "P99_CAR": p99,
            "N_original": len(data),
            "N_final": len(data_out),
        })

    # --------------------------------------------------------
    # E. ROBUSTEZ: DIVIDEND YIELD WINSORIZADO 1% / 1%
    #    scipy.mstats.winsorize replica el PTG: max ~0.165.
    # --------------------------------------------------------
    data_w = data.copy()
    data_w["DY_winsor"] = np.asarray(
        winsorize(data_w["DY"].to_numpy(dtype=float), limits=[0.01, 0.01]),
        dtype=float,
    )

    for y, ventana in [("CAR_0_1", "[0,+1]"), ("CAR_0_2", "[0,+2]")]:
        m_win = ajustar_ols(f"{y} ~ DY_winsor + C(Anio_FE)", data_w, hc3=True)

        resultados_full.append(
            extraer_modelo(m_win, "DY_winsor_HC3_FE", ventana, "DY_winsor")
        )
        resultados_clave.append(
            fila_clave(m_win, "DY_winsor_HC3_FE", ventana, "DY_winsor")
        )

    regresiones_full = pd.concat(resultados_full, ignore_index=True)
    regresiones_clave = pd.DataFrame(resultados_clave)
    outliers_info = pd.DataFrame(info_outliers)

    winsor_info = pd.DataFrame([{
        "DY_min_original": data["DY"].min(),
        "DY_max_original": data["DY"].max(),
        "DY_min_winsor": data_w["DY_winsor"].min(),
        "DY_max_winsor": data_w["DY_winsor"].max(),
        "N": len(data_w),
    }])

    return regresiones_full, regresiones_clave, outliers_info, winsor_info


# ============================================================
# 14. EJECUTAR TODO
# ============================================================

df_final = df.copy()

# AR anuncio y pago
for evento in ["anuncio", "pago"]:
    df_final = crear_AR(df_final, evento=evento, ventana=2)

# CAR usados en PTG y presentacion
for evento in ["anuncio", "pago"]:
    for inicio, fin in [(-1, 1), (-2, 2), (0, 1), (0, 2)]:
        df_final = crear_CAR_rango(
            df_final,
            evento=evento,
            inicio=inicio,
            fin=fin,
        )

# Clasificacion por tipo sobre la muestra total
df_final = crear_tipo_dividendo_agrupado(df_final)

# ------------------------------------------------------------
# MUESTRA PRINCIPAL: 633 REGULAR CASH
# ------------------------------------------------------------
df_regular = df_final.loc[df_final["Type"].eq("Regular Cash")].copy()

df_regular = crear_grupos_dividend_yield(
    df_regular, q=3, nombre_columna="Grupo_Dividend_Yield_Q3"
)
df_regular = crear_grupos_dividend_yield(
    df_regular, q=4, nombre_columna="Grupo_Dividend_Yield_Q4"
)
df_regular = crear_grupos_dividend_yield(
    df_regular, q=5, nombre_columna="Grupo_Dividend_Yield_Q5"
)

df_regular = agregar_variables_regresion(df_regular)

print("============================================================")
print(f"Muestra total                 : {len(df_final)}")
print(f"Muestra principal Regular Cash: {len(df_regular)}")
print("============================================================")

if len(df_final) != 774:
    print("ADVERTENCIA: la muestra total no tiene 774 eventos.")
if len(df_regular) != 633:
    print("ADVERTENCIA: Regular Cash no tiene 633 eventos.")


# ============================================================
# 15. RESULTADOS DESCRIPTIVOS Y TESTS
# ============================================================

# Principal: 633
resumen_principal = crear_resumen_descriptivo(df_regular)
tests_principales = crear_tests_significatividad(df_regular)
tests_robustos_principales = crear_tests_robustos(df_regular)

# Complementario: 774
resumen_total = crear_resumen_descriptivo(df_final)
tests_total = crear_tests_significatividad(df_final)
tests_robustos_total = crear_tests_robustos(df_final)

# Por tipo: 774
resumen_tipo = crear_resumen_por_grupo(df_final, "Tipo_Dividendo_Agrupado")
tests_tipo = crear_tests_por_grupo(df_final, "Tipo_Dividendo_Agrupado")
tests_robustos_tipo = crear_tests_robustos_por_grupo(
    df_final, "Tipo_Dividendo_Agrupado"
)

# Yield: 633
resumen_yield_q3 = crear_resumen_por_grupo(df_regular, "Grupo_Dividend_Yield_Q3")
tests_yield_q3 = crear_tests_por_grupo(df_regular, "Grupo_Dividend_Yield_Q3")
tests_robustos_yield_q3 = crear_tests_robustos_por_grupo(
    df_regular, "Grupo_Dividend_Yield_Q3"
)

resumen_yield_q4 = crear_resumen_por_grupo(df_regular, "Grupo_Dividend_Yield_Q4")
tests_yield_q4 = crear_tests_por_grupo(df_regular, "Grupo_Dividend_Yield_Q4")
tests_robustos_yield_q4 = crear_tests_robustos_por_grupo(
    df_regular, "Grupo_Dividend_Yield_Q4"
)

resumen_yield_q5 = crear_resumen_por_grupo(df_regular, "Grupo_Dividend_Yield_Q5")
tests_yield_q5 = crear_tests_por_grupo(df_regular, "Grupo_Dividend_Yield_Q5")
tests_robustos_yield_q5 = crear_tests_robustos_por_grupo(
    df_regular, "Grupo_Dividend_Yield_Q5"
)


# ============================================================
# 16. TABLAS AUXILIARES DE PRESENTACION
# ============================================================

aar_anuncio_633 = crear_aar_diario(df_regular, "anuncio")
aar_pago_633 = crear_aar_diario(df_regular, "pago")
caar_anuncio_633 = crear_caar_desde_menos1(df_regular, "anuncio")
caar_pago_633 = crear_caar_desde_menos1(df_regular, "pago")

# Se conserva tambien la version 774 como comparacion.
aar_anuncio_774 = crear_aar_diario(df_final, "anuncio")
aar_pago_774 = crear_aar_diario(df_final, "pago")

binscatter_20 = crear_binscatter(df_regular, n_bins=20)

# Composicion
composicion_tipo = (
    df_final["Tipo_Dividendo_Agrupado"]
    .value_counts(dropna=False)
    .rename_axis("Tipo")
    .reset_index(name="Eventos")
)
composicion_tipo["Participacion_%"] = composicion_tipo["Eventos"] / len(df_final) * 100

# Cobertura por ano calendario correctamente parseado para descripcion.
fechas_total = df_final["Fecha de aviso"].map(convertir_fecha_excel_mixta)
cobertura_anual = (
    fechas_total.dt.year
    .value_counts()
    .sort_index()
    .rename_axis("Ano")
    .reset_index(name="Eventos")
)

# Cobertura sectorial
cobertura_sectorial = (
    df_final["Sector"]
    .value_counts(dropna=False)
    .rename_axis("Sector")
    .reset_index(name="Eventos")
)

# Dividend Yield descriptivo y cortes
q_cuts = df_regular["DY"].quantile([0.25, 0.50, 0.75])
yield_resumen = pd.DataFrame([{
    "N": df_regular["DY"].notna().sum(),
    "Media": df_regular["DY"].mean(),
    "Mediana": df_regular["DY"].median(),
    "Minimo": df_regular["DY"].min(),
    "Maximo": df_regular["DY"].max(),
    "Q1_corte_25%": q_cuts.loc[0.25],
    "Q2_corte_50%": q_cuts.loc[0.50],
    "Q3_corte_75%": q_cuts.loc[0.75],
}])

# Diagnostico de anos FE
conteo_anio_ptg = (
    df_regular["Anio_FE_PTG"].value_counts(dropna=False).sort_index()
    .rename_axis("Anio_FE_PTG").reset_index(name="N")
)
conteo_anio_correcto = (
    df_regular["Anio_calendario"].value_counts(dropna=False).sort_index()
    .rename_axis("Anio_calendario").reset_index(name="N")
)


# ============================================================
# 17. REGRESIONES Y ROBUSTEZ
# ============================================================

regresiones_full, regresiones_clave, outliers_info, winsor_info = correr_regresiones(
    df_regular
)


# ============================================================
# 18. CHEQUEOS CONTRA RESULTADOS CENTRALES DEL PTG
# ============================================================

def buscar_coef(df_reg, modelo, ventana, variable):
    fila = df_reg.loc[
        (df_reg["Modelo"] == modelo)
        & (df_reg["Ventana"] == ventana)
        & (df_reg["Variable"] == variable)
    ]
    if fila.empty:
        return np.nan, np.nan
    return float(fila.iloc[0]["Coeficiente"]), float(fila.iloc[0]["P_value"])


checks = []

# H1 principal con 633 Regular Cash
h1 = tests_robustos_principales.loc[
    tests_robustos_principales["Variable"] == "AR_anuncio_+0"
].iloc[0]
checks.append({
    "Chequeo": "H1_AR0_633",
    "Valor": h1["Media_%"],
    "Esperado_aprox": 0.413,
    "Detalle": f"p_t={h1['p_t']:.8f}; p_w={h1['p_wilcoxon']:.8f}; p_sign={h1['p_sign']:.8f}",
})

# Continuo HC3
b, p = buscar_coef(regresiones_clave, "DY_continuo_HC3", "[0,+1]", "DY")
checks.append({
    "Chequeo": "DY_continuo_0_1",
    "Valor": b,
    "Esperado_aprox": 0.2349,
    "Detalle": f"p={p:.6f}",
})

# Continuo HC3 + FE
b, p = buscar_coef(regresiones_clave, "DY_continuo_HC3_FE", "[0,+1]", "DY")
checks.append({
    "Chequeo": "DY_continuo_FE_0_1",
    "Valor": b,
    "Esperado_aprox": 0.2396 if REPRODUCIR_FE_PTG else np.nan,
    "Detalle": f"p={p:.6f}",
})

# Winsor
b, p = buscar_coef(regresiones_clave, "DY_winsor_HC3_FE", "[0,+1]", "DY_winsor")
checks.append({
    "Chequeo": "DY_winsor_0_1",
    "Valor": b,
    "Esperado_aprox": 0.2526 if REPRODUCIR_FE_PTG else np.nan,
    "Detalle": f"p={p:.6f}",
})

chequeos_ptg = pd.DataFrame(checks)


# ============================================================
# 19. EXPORTAR EXCEL COMPLETO
# ============================================================

with pd.ExcelWriter(salida, engine="openpyxl") as writer:
    # Bases
    df_final.to_excel(writer, sheet_name="Base_total_774", index=False)
    df_regular.to_excel(writer, sheet_name="Base_RegularCash_633", index=False)

    # Principal 633
    resumen_principal.to_excel(writer, sheet_name="Resumen_principal_633")
    tests_principales.to_excel(writer, sheet_name="Tests_principales_633", index=False)
    tests_robustos_principales.to_excel(writer, sheet_name="Tests_robustos_633", index=False)

    # Complementario 774
    resumen_total.to_excel(writer, sheet_name="Resumen_total_774")
    tests_total.to_excel(writer, sheet_name="Tests_total_774", index=False)
    tests_robustos_total.to_excel(writer, sheet_name="Tests_robustos_774", index=False)

    # Tipo de dividendo
    composicion_tipo.to_excel(writer, sheet_name="Composicion_tipo", index=False)
    resumen_tipo.to_excel(writer, sheet_name="Resumen_tipo_774")
    tests_tipo.to_excel(writer, sheet_name="Tests_tipo_774", index=False)
    tests_robustos_tipo.to_excel(writer, sheet_name="Robustos_tipo_774", index=False)

    # Yield Q3/Q4/Q5 sobre 633
    resumen_yield_q3.to_excel(writer, sheet_name="Resumen_Q3_633")
    tests_yield_q3.to_excel(writer, sheet_name="Tests_Q3_633", index=False)
    tests_robustos_yield_q3.to_excel(writer, sheet_name="Robustos_Q3_633", index=False)

    resumen_yield_q4.to_excel(writer, sheet_name="Resumen_Q4_633")
    tests_yield_q4.to_excel(writer, sheet_name="Tests_Q4_633", index=False)
    tests_robustos_yield_q4.to_excel(writer, sheet_name="Robustos_Q4_633", index=False)

    resumen_yield_q5.to_excel(writer, sheet_name="Resumen_Q5_633")
    tests_yield_q5.to_excel(writer, sheet_name="Tests_Q5_633", index=False)
    tests_robustos_yield_q5.to_excel(writer, sheet_name="Robustos_Q5_633", index=False)

    # Presentacion
    pd.concat([aar_anuncio_633, aar_pago_633], ignore_index=True).to_excel(
        writer, sheet_name="AAR_diario_633", index=False
    )
    pd.concat([caar_anuncio_633, caar_pago_633], ignore_index=True).to_excel(
        writer, sheet_name="CAAR_menos1_633", index=False
    )
    pd.concat([aar_anuncio_774, aar_pago_774], ignore_index=True).to_excel(
        writer, sheet_name="AAR_diario_774", index=False
    )
    binscatter_20.to_excel(writer, sheet_name="Binscatter_20_633", index=False)
    yield_resumen.to_excel(writer, sheet_name="Yield_resumen_633", index=False)
    cobertura_anual.to_excel(writer, sheet_name="Cobertura_anual", index=False)
    cobertura_sectorial.to_excel(writer, sheet_name="Cobertura_sector", index=False)

    # Regresiones
    regresiones_clave.to_excel(writer, sheet_name="Regresiones_clave", index=False)
    regresiones_full.to_excel(writer, sheet_name="Regresiones_full", index=False)
    outliers_info.to_excel(writer, sheet_name="Outliers_p1_p99", index=False)
    winsor_info.to_excel(writer, sheet_name="Winsor_info", index=False)

    # Diagnostico FE
    conteo_anio_ptg.to_excel(writer, sheet_name="FE_anos_PTG", index=False)
    conteo_anio_correcto.to_excel(writer, sheet_name="FE_anos_correctos", index=False)

    # Chequeos rapidos
    chequeos_ptg.to_excel(writer, sheet_name="Chequeos_PTG", index=False)


# ============================================================
# 20. MOSTRAR RESULTADOS CLAVE EN CONSOLA
# ============================================================

print("\nRESULTADO PRINCIPAL H1 - REGULAR CASH (N=633)")
print(
    tests_robustos_principales.loc[
        tests_robustos_principales["Variable"] == "AR_anuncio_+0",
        [
            "N", "Media_%", "Mediana_%", "%_positivos",
            "t_stat", "p_t", "p_wilcoxon", "p_sign",
            "p_bootstrap", "IC95_low_%", "IC95_high_%",
        ],
    ].to_string(index=False)
)

print("\nREGRESIONES CLAVE")
print(
    regresiones_clave[
        ["Modelo", "Ventana", "Variable", "Coeficiente", "P_value", "R2", "N"]
    ].to_string(index=False)
)

print("\nWINSORIZACION")
print(winsor_info.to_string(index=False))

print("\nOUTLIERS p1/p99")
print(outliers_info.to_string(index=False))

print("\nCHEQUEO DE ANOS FE")
print(f"REPRODUCIR_FE_PTG = {REPRODUCIR_FE_PTG}")
print("Distribucion Anio_FE_PTG:")
print(conteo_anio_ptg.to_string(index=False))

print(f"\nExcel creado en: {salida}")
