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

Las secciones 21 a 24 incorporan los dos bloques agregados para la defensa
final: la dinamica de la ventana de fecha de pago y el modelo naive de
expectativas de dividendo. Ninguno de los dos altera los resultados de las
secciones 1 a 20; se agregan como hojas nuevas del mismo Excel de salida.

Requisitos: Python 3.10+, pandas, numpy, scipy, statsmodels, openpyxl.
Los datos de entrada (Bloomberg, serie PX_LAST) no se distribuyen en este
repositorio por restricciones de licencia.
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

# False = especificacion final: usa el ano calendario correctamente parseado (1996-2026).
# True = solo diagnostico legacy para reproducir los FE de la version originalmente enviada.
REPRODUCIR_FE_PTG = False

N_BOOT = 10000
SEED_BOOT = 42

# Ventana del pronostico naive: se considera que la empresa "pago el ano
# anterior" si anuncio otro dividendo ordinario dentro de estos dias previos.
DIAS_VENTANA_FORECAST = 365


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
            "Pasa_4_pruebas": np.nan,
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

    # 5. Criterio conjunto: las cuatro pruebas significativas al 5% y el
    #    intervalo bootstrap excluyendo el cero. Es el criterio de lectura
    #    declarado en el trabajo: si las cuatro coinciden, la conclusion no
    #    depende del supuesto estadistico elegido.
    pasa_4 = int(
        (p_t == p_t and p_t < 0.05)
        and (p_w == p_w and p_w < 0.05)
        and (p_sign == p_sign and p_sign < 0.05)
        and not (ic_low <= 0 <= ic_high)
    )

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
        "Pasa_4_pruebas": pasa_4,
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


def crear_tests_robustos_variables(df, variables, etiquetas=None):
    """Bateria de cuatro pruebas sobre una lista arbitraria de columnas.

    Complementa a crear_tests_robustos, que recorre siempre las variables
    de obtener_variables_resultado.
    """
    resultados = []

    for i, var in enumerate(variables):
        fila = {"Variable": etiquetas[i] if etiquetas else var}
        fila.update(_tests_una_serie(df[var]))
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

def ajustar_ols(formula, data, hc3=False, cluster=None):
    """MCO clasico, con matriz robusta HC3 o con errores clusterizados.

    cluster: nombre de la columna de agrupamiento, por ejemplo "Company".
    Los eventos de una misma empresa no son independientes entre si, de modo
    que el clustering relaja el supuesto de independencia que HC3 mantiene.
    Modifica los errores estandar, no los coeficientes.
    """
    if cluster is not None:
        modelo = smf.ols(formula, data=data)
        grupos = data.loc[modelo.data.row_labels, cluster]
        return modelo.fit(
            cov_type="cluster",
            cov_kwds={"groups": grupos},
            use_t=False,
        )

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


def fila_modelo(modelo, nombre_modelo, variables, wald=None):
    """Resumen compacto de un modelo: intercepto, variables pedidas y, de
    manera opcional, el p-valor de una restriccion lineal (test de Wald)."""
    fila = {
        "Modelo": nombre_modelo,
        "N": int(modelo.nobs),
        "R2": modelo.rsquared,
        "Intercepto": modelo.params.get("Intercept", np.nan),
        "p_Intercepto": modelo.pvalues.get("Intercept", np.nan),
    }

    for variable in variables:
        fila[f"b_{variable}"] = modelo.params.get(variable, np.nan)
        fila[f"ee_{variable}"] = modelo.bse.get(variable, np.nan)
        fila[f"p_{variable}"] = modelo.pvalues.get(variable, np.nan)

    if wald is not None:
        fila["Restriccion_Wald"] = wald
        fila["p_Wald"] = float(np.squeeze(modelo.wald_test(wald, use_f=False).pvalue))

    return fila


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

    # --------------------------------------------------------
    # F. ROBUSTEZ: ERRORES ESTANDAR CLUSTERIZADOS POR EMPRESA
    #    La muestra tiene 633 eventos de 43 empresas, de modo que las
    #    observaciones de una misma empresa no son independientes. Los
    #    coeficientes son identicos a los de HC3; cambian los errores.
    # --------------------------------------------------------
    for y, ventana in [("CAR_0_1", "[0,+1]"), ("CAR_0_2", "[0,+2]")]:
        m_clu = ajustar_ols(f"{y} ~ DY", data, cluster="Company")
        m_clu_fe = ajustar_ols(f"{y} ~ DY + C(Anio_FE)", data, cluster="Company")
        m_clu_q = ajustar_ols(
            f'{y} ~ C(DY_Q4, Treatment(reference="Q1"))', data, cluster="Company"
        )

        nombre_q4 = 'C(DY_Q4, Treatment(reference="Q1"))[T.Q4]'

        resultados_full.extend([
            extraer_modelo(m_clu, "DY_continuo_Cluster", ventana, "DY"),
            extraer_modelo(m_clu_fe, "DY_continuo_Cluster_FE", ventana, "DY"),
            extraer_modelo(m_clu_q, "Cuartiles_Cluster", ventana, nombre_q4),
        ])

        resultados_clave.extend([
            fila_clave(m_clu, "DY_continuo_Cluster", ventana, "DY"),
            fila_clave(m_clu_fe, "DY_continuo_Cluster_FE", ventana, "DY"),
            fila_clave(m_clu_q, "Cuartiles_Cluster", ventana, nombre_q4),
        ])

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
# 19. VENTANA DE FECHA DE PAGO
#     Bloque agregado para la defensa final. Documenta la dinamica del
#     patron de caida y rebote alrededor de la fecha de pago.
# ============================================================

def crear_AR_ex_date(df):
    """AR alrededor de la ex date.

    Se construye con las mismas columnas de retorno de la base que usan los
    AR de anuncio y de pago: retorno de la accion menos retorno del indice.
    """
    df = df.copy()

    pares = {
        -1: ("Retorno -1d", "Retorno -1d.1"),
         0: ("Retorno ex_date", "Retorno ex_date.1"),
         1: ("Retorno +1D", "Retorno +1D.1"),
    }

    for dia, (retorno_accion, retorno_mercado) in pares.items():
        df[f"AR_ex_{dia:+d}"] = (
            pd.to_numeric(df[retorno_accion], errors="coerce")
            - pd.to_numeric(df[retorno_mercado], errors="coerce")
        )

    return df


def preparar_datos_ventanas(df):
    """Copia con nombres simples para las formulas de statsmodels.

    Los nombres originales llevan signos y corchetes, que patsy interpreta
    como operadores.
    """
    equivalencias = {
        "AR_anuncio_+0": "AR_anu_0",
        "AR_anuncio_+1": "AR_anu_1",
        "AR_pago_-1": "AR_pago_m1",
        "AR_pago_+0": "AR_pago_0",
        "AR_pago_+1": "AR_pago_1",
        "AR_pago_+2": "AR_pago_2",
        "CAR_anuncio_[0,+1]": "CAR_anu_01",
        "CAR_anuncio_[0,+2]": "CAR_anu_02",
        "CAR_pago_[0,+1]": "CAR_pago_01",
        "CAR_pago_[0,+2]": "CAR_pago_02",
        "CAR_pago_[-1,+1]": "CAR_pago_m11",
        "AR_ex_-1": "AR_ex_m1",
        "AR_ex_+0": "AR_ex_0",
        "AR_ex_+1": "AR_ex_1",
    }

    datos = df.copy()
    for original, simple in equivalencias.items():
        if original in datos.columns:
            datos[simple] = pd.to_numeric(datos[original], errors="coerce")

    return datos


def crear_descomposicion_reversion(datos):
    """Separa el rebote del dia +1 en reversion y drift.

    El intercepto mide la parte del AR(+1) que no se explica por la caida del
    dia anterior; el coeficiente negativo mide la reversion. La ventana de
    anuncio se estima como placebo: si alli el intercepto no es significativo,
    el patron es propio de la fecha de pago y no una regularidad general de
    estas acciones.
    """
    m_pago = ajustar_ols("AR_pago_1 ~ AR_pago_0", datos, hc3=True)
    m_anuncio = ajustar_ols("AR_anu_1 ~ AR_anu_0", datos, hc3=True)

    tabla = pd.DataFrame([
        fila_modelo(m_pago, "Pago: AR(+1) ~ AR(0)", ["AR_pago_0"]),
        fila_modelo(m_anuncio, "Anuncio (placebo): AR(+1) ~ AR(0)", ["AR_anu_0"]),
    ])

    tabla["Nota"] = [
        "Intercepto = drift no explicado por reversion; coeficiente < 0 = reversion",
        "Placebo: intercepto no significativo implica patron propio del pago",
    ]

    return tabla


def crear_escala_dividendo_pago(datos):
    """La caida y el rebote, contra el tamano del dividendo.

    Si el patron respondiera a la reinversion del efectivo cobrado o a la
    venta de la clientela de dividendos, deberia crecer con el yield.
    """
    filas = []

    for variable, etiqueta in [
        ("AR_pago_0", "AR pago (0) ~ DY"),
        ("AR_pago_1", "AR pago (+1) ~ DY"),
        ("CAR_pago_01", "CAR pago [0,+1] ~ DY"),
    ]:
        modelo = ajustar_ols(f"{variable} ~ DY", datos, hc3=True)
        filas.append(fila_modelo(modelo, etiqueta, ["DY"]))

    return pd.DataFrame(filas)


def crear_cortes_ventana_pago(datos, columna_grupo, etiqueta):
    """Media, p-valor y correlacion AR(0)-AR(+1) dentro de cada grupo."""
    filas = []

    for grupo, sub in datos.groupby(columna_grupo, observed=False):
        if len(sub) == 0:
            continue

        fila = {"Corte": etiqueta, "Grupo": str(grupo), "N": len(sub)}

        for variable, nombre in [
            ("AR_pago_0", "AR(0)"),
            ("AR_pago_1", "AR(+1)"),
            ("CAR_pago_01", "CAR[0,+1]"),
        ]:
            serie = sub[variable].dropna()
            t_stat, p_val = ttest_1samp(serie, 0) if len(serie) > 1 else (np.nan, np.nan)
            fila[f"{nombre}_%"] = serie.mean() * 100
            fila[f"{nombre}_p"] = p_val

        par = sub[["AR_pago_0", "AR_pago_1"]].dropna()
        fila["corr_AR0_AR1"] = par.corr().iloc[0, 1] if len(par) > 2 else np.nan

        filas.append(fila)

    return pd.DataFrame(filas)


# ------------------------------------------------------------
# EJECUCION DEL BLOQUE DE FECHA DE PAGO
# ------------------------------------------------------------

df_regular = crear_AR_ex_date(df_regular)
datos_ventanas = preparar_datos_ventanas(df_regular)

# Cuatro pruebas sobre cada dia de la ventana de pago.
pago_cuatro_pruebas = crear_tests_robustos_variables(
    datos_ventanas,
    ["AR_pago_m1", "AR_pago_0", "AR_pago_1", "AR_pago_2",
     "CAR_pago_01", "CAR_pago_02", "CAR_pago_m11"],
    ["AR pago (-1)", "AR pago (0)", "AR pago (+1)", "AR pago (+2)",
     "CAR pago [0,+1]", "CAR pago [0,+2]", "CAR pago [-1,+1]"],
)

# Misma bateria sobre la ventana de anuncio, para comparar.
anuncio_cuatro_pruebas = crear_tests_robustos_variables(
    datos_ventanas,
    ["AR_anu_0", "AR_anu_1", "CAR_anu_01"],
    ["AR anuncio (0)", "AR anuncio (+1)", "CAR anuncio [0,+1]"],
)

pago_descomposicion = crear_descomposicion_reversion(datos_ventanas)
pago_escala_dividendo = crear_escala_dividendo_pago(datos_ventanas)

# Cortes: magnitud del dividendo, subperiodo y liquidez.
datos_ventanas["Q_DY"] = pd.qcut(
    datos_ventanas["DY"], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop"
)
datos_ventanas["Subperiodo"] = pd.cut(
    datos_ventanas["Anio_calendario"],
    bins=[1995, 2009, 2017, 2026],
    labels=["1996-2009", "2010-2017", "2018-2026"],
)
empresas_frecuentes = datos_ventanas["Company"].value_counts().head(6).index.tolist()
datos_ventanas["Tamano"] = np.where(
    datos_ventanas["Company"].isin(empresas_frecuentes),
    "6 con mas eventos",
    "Resto",
)

pago_cortes = pd.concat([
    crear_cortes_ventana_pago(datos_ventanas, "Q_DY", "Cuartil de Dividend Yield"),
    crear_cortes_ventana_pago(datos_ventanas, "Subperiodo", "Subperiodo"),
    crear_cortes_ventana_pago(datos_ventanas, "Tamano", "Tamano / liquidez"),
], ignore_index=True)

pago_cortes_nota = pd.DataFrame([{
    "Empresas_mas_frecuentes": ", ".join(empresas_frecuentes),
    "Criterio": "seis empresas con mas anuncios dentro de la muestra Regular Cash",
}])

# Ex date: se reporta como respaldo de la ventana de pago.
ex_date_cuatro_pruebas = crear_tests_robustos_variables(
    datos_ventanas,
    ["AR_ex_m1", "AR_ex_0", "AR_ex_1"],
    ["AR ex date (-1)", "AR ex date (0)", "AR ex date (+1)"],
)

ex_date_ajuste = pd.DataFrame([
    fila_modelo(
        ajustar_ols("AR_ex_0 ~ DY", datos_ventanas, hc3=True),
        "AR ex (0) ~ DY  [drop-off ratio]",
        ["DY"],
    ),
    fila_modelo(
        ajustar_ols("AR_ex_m1 ~ DY", datos_ventanas, hc3=True),
        "AR ex (-1) ~ DY",
        ["DY"],
    ),
])


# ============================================================
# 20. MODELO NAIVE DE EXPECTATIVAS DE DIVIDENDO
#     Bloque agregado para la defensa final. Contrasta la parte esperada
#     del dividendo contra la sorpresa respecto de un pronostico ingenuo.
# ============================================================

def construir_pronostico_naive(df, dias=DIAS_VENTANA_FORECAST):
    """Pronostico ingenuo del dividendo del anuncio.

    Regla: si la empresa anuncio otro dividendo ordinario dentro de los dias
    previos indicados, se espera el mismo monto por accion; si no pago en ese
    lapso, el pronostico es cero. Se construyen dos versiones de la sorpresa:

      - en yield, con el ultimo monto valuado al precio del anuncio, que es
        la unidad comparable a lo largo de treinta anos de inflacion;
      - en dividendo nominal, que se reporta solo como contraste.
    """
    datos = df.copy()
    datos["DPA"] = pd.to_numeric(datos["Ammount"], errors="coerce")
    datos["P0"] = pd.to_numeric(datos["Precio_fecha de aviso"], errors="coerce")
    datos = datos.sort_values(["Company", "Fecha_aviso_corregida"]).reset_index(drop=True)

    dpa_previo, n_previos, dias_previos = [], [], []

    for _, fila in datos.iterrows():
        fecha = fila["Fecha_aviso_corregida"]

        previos = datos.loc[
            (datos["Company"] == fila["Company"])
            & (datos["Fecha_aviso_corregida"] < fecha)
            & (datos["Fecha_aviso_corregida"] >= fecha - pd.Timedelta(days=dias))
        ]

        n_previos.append(len(previos))

        if len(previos) == 0:
            dpa_previo.append(0.0)
            dias_previos.append(np.nan)
        else:
            ultimo = previos.sort_values("Fecha_aviso_corregida").iloc[-1]
            dpa_previo.append(float(ultimo["DPA"]))
            dias_previos.append((fecha - ultimo["Fecha_aviso_corregida"]).days)

    datos["DPA_previo"] = dpa_previo
    datos["N_previos_12m"] = n_previos
    datos["Dias_desde_previo"] = dias_previos
    datos["Sin_pago_previo"] = (datos["N_previos_12m"] == 0).astype(int)

    # Version en yield
    datos["DY_esperado"] = datos["DPA_previo"] / datos["P0"]
    datos["Sorpresa_DY"] = datos["DY"] - datos["DY_esperado"]

    # Version en dividendo nominal
    datos["DPA_esperado"] = datos["DPA_previo"]
    datos["Sorpresa_DPA"] = datos["DPA"] - datos["DPA_esperado"]

    return datos


def crear_calidad_pronostico(realizado, pronosticado, etiqueta, unidad):
    """Poder predictivo del pronostico naive sobre el dividendo anunciado."""
    aux = pd.DataFrame({"y": realizado, "x": pronosticado}).dropna()

    if len(aux) < 3:
        return {}

    error = aux["y"] - aux["x"]
    modelo = smf.ols("y ~ x", data=aux).fit()

    return {
        "Pronostico": etiqueta,
        "Unidad": unidad,
        "N": len(aux),
        "R2": modelo.rsquared,
        "Pendiente": modelo.params["x"],
        "p_Pendiente": modelo.pvalues["x"],
        "RMSE": np.sqrt((error ** 2).mean()),
        "MAE": error.abs().mean(),
        "Sesgo_medio": error.mean(),
    }


def crear_descomposicion_expectativas(datos, y, esperado, sorpresa, etiqueta,
                                      extra=None, cluster=None):
    """CAR contra la parte esperada y la sorpresa del dividendo.

    Como el yield anunciado es por construccion la suma de ambas, el test de
    Wald de igualdad de coeficientes responde si la descomposicion agrega algo
    sobre la regresion del CAR contra el yield total.
    """
    formula = f"{y} ~ {esperado} + {sorpresa}"
    if extra:
        formula = f"{formula} + {extra}"

    if cluster is not None:
        modelo = ajustar_ols(formula, datos, cluster=cluster)
    else:
        modelo = ajustar_ols(formula, datos, hc3=True)

    variables = [esperado, sorpresa] + ([extra] if extra else [])
    fila = fila_modelo(modelo, etiqueta, variables, wald=f"{esperado} = {sorpresa}")
    fila["Variable_dependiente"] = y
    fila["Errores"] = "Cluster por empresa" if cluster else "HC3"

    return fila


def crear_signo_sorpresa(valor, tolerancia=0.001):
    """Clasifica el anuncio segun supere, iguale o no alcance el pronostico."""
    if pd.isna(valor):
        return np.nan
    if valor > tolerancia:
        return "Sorpresa positiva"
    if valor < -tolerancia:
        return "Sorpresa negativa"
    return "Sin sorpresa (+-0,1pp)"


# ------------------------------------------------------------
# EJECUCION DEL BLOQUE DE EXPECTATIVAS
# ------------------------------------------------------------

df_forecast = construir_pronostico_naive(datos_ventanas)
df_forecast_con_historia = df_forecast.loc[df_forecast["Sin_pago_previo"] == 0].copy()

forecast_cobertura = pd.DataFrame([{
    "N_total": len(df_forecast),
    "N_con_pago_previo_12m": int((df_forecast["Sin_pago_previo"] == 0).sum()),
    "N_sin_pago_previo_12m": int((df_forecast["Sin_pago_previo"] == 1).sum()),
    "Dias_medios_al_pago_previo": np.nanmean(df_forecast["Dias_desde_previo"]),
    "Ventana_dias": DIAS_VENTANA_FORECAST,
}])

forecast_calidad = pd.DataFrame([
    crear_calidad_pronostico(
        df_forecast["DY"], df_forecast["DY_esperado"],
        "Yield - toda la muestra (0 si no pago)", "yield",
    ),
    crear_calidad_pronostico(
        df_forecast_con_historia["DY"], df_forecast_con_historia["DY_esperado"],
        "Yield - solo con pago previo", "yield",
    ),
    crear_calidad_pronostico(
        df_forecast["DPA"], df_forecast["DPA_esperado"],
        "Nominal - toda la muestra (0 si no pago)", "pesos por accion",
    ),
    crear_calidad_pronostico(
        df_forecast_con_historia["DPA"], df_forecast_con_historia["DPA_esperado"],
        "Nominal - solo con pago previo", "pesos por accion",
    ),
])

filas_expectativas = []

# Referencia: la regresion del CAR contra el yield total.
filas_expectativas.append(
    fila_modelo(
        ajustar_ols("CAR_anu_01 ~ DY", df_forecast, hc3=True),
        "REFERENCIA: CAR[0,+1] ~ DY", ["DY"],
    )
)
filas_expectativas.append(
    fila_modelo(
        ajustar_ols("CAR_anu_02 ~ DY", df_forecast, hc3=True),
        "REFERENCIA: CAR[0,+2] ~ DY", ["DY"],
    )
)

# Descomposicion en yield.
filas_expectativas.extend([
    crear_descomposicion_expectativas(
        df_forecast, "CAR_anu_01", "DY_esperado", "Sorpresa_DY",
        "Yield: CAR[0,+1], toda la muestra",
    ),
    crear_descomposicion_expectativas(
        df_forecast, "CAR_anu_02", "DY_esperado", "Sorpresa_DY",
        "Yield: CAR[0,+2], toda la muestra",
    ),
    crear_descomposicion_expectativas(
        df_forecast, "AR_anu_0", "DY_esperado", "Sorpresa_DY",
        "Yield: AR(0), toda la muestra",
    ),
    crear_descomposicion_expectativas(
        df_forecast, "CAR_anu_01", "DY_esperado", "Sorpresa_DY",
        "Yield: CAR[0,+1], con efectos fijos por ano", extra="C(Anio_FE)",
    ),
    crear_descomposicion_expectativas(
        df_forecast, "CAR_anu_01", "DY_esperado", "Sorpresa_DY",
        "Yield: CAR[0,+1], con dummy de sin pago previo", extra="Sin_pago_previo",
    ),
    crear_descomposicion_expectativas(
        df_forecast, "CAR_anu_01", "DY_esperado", "Sorpresa_DY",
        "Yield: CAR[0,+1], cluster por empresa", cluster="Company",
    ),
    crear_descomposicion_expectativas(
        df_forecast_con_historia, "CAR_anu_01", "DY_esperado", "Sorpresa_DY",
        "Yield: CAR[0,+1], solo con pago previo",
    ),
    crear_descomposicion_expectativas(
        df_forecast_con_historia, "CAR_anu_01", "DY_esperado", "Sorpresa_DY",
        "Yield: CAR[0,+1], solo con pago previo, cluster", cluster="Company",
    ),
    crear_descomposicion_expectativas(
        df_forecast_con_historia, "CAR_anu_02", "DY_esperado", "Sorpresa_DY",
        "Yield: CAR[0,+2], solo con pago previo",
    ),
])

# Descomposicion en dividendo nominal. Se reporta como contraste: con treinta
# anos de inflacion los montos no son comparables entre si y el ajuste del
# modelo recoge, en buena medida, el nivel de precios de cada ano.
filas_expectativas.extend([
    crear_descomposicion_expectativas(
        df_forecast, "CAR_anu_01", "DPA_esperado", "Sorpresa_DPA",
        "Nominal: CAR[0,+1], toda la muestra",
    ),
    crear_descomposicion_expectativas(
        df_forecast_con_historia, "CAR_anu_01", "DPA_esperado", "Sorpresa_DPA",
        "Nominal: CAR[0,+1], solo con pago previo",
    ),
])

expectativas_regresiones = pd.DataFrame(filas_expectativas)

# Reaccion segun el signo de la sorpresa.
filas_signo = []

for muestra, etiqueta in [
    (df_forecast, "Toda la muestra"),
    (df_forecast_con_historia, "Solo con pago previo"),
]:
    aux = muestra.copy()
    aux["Signo_sorpresa"] = aux["Sorpresa_DY"].map(crear_signo_sorpresa)

    for grupo, sub in aux.groupby("Signo_sorpresa", observed=False):
        for variable, nombre in [("AR_anu_0", "AR(0)"), ("CAR_anu_01", "CAR[0,+1]")]:
            # El Dividend Yield medio de cada grupo permite distinguir si la
            # reaccion responde a la sorpresa o al nivel del dividendo.
            fila = {
                "Muestra": etiqueta,
                "Grupo": grupo,
                "DY_medio_%": sub["DY"].mean() * 100,
                "Variable": nombre,
            }
            fila.update(_tests_una_serie(sub[variable]))
            filas_signo.append(fila)

expectativas_signo = pd.DataFrame(filas_signo)

# Cuartiles de sorpresa.
df_forecast["Q_sorpresa"] = pd.qcut(
    df_forecast["Sorpresa_DY"], 4, labels=["S1", "S2", "S3", "S4"], duplicates="drop"
)

expectativas_cuartiles = (
    df_forecast.groupby("Q_sorpresa", observed=False)
    .agg(
        N=("CAR_anu_01", "size"),
        Sorpresa_media=("Sorpresa_DY", "mean"),
        DY_medio=("DY", "mean"),
        AR_0=("AR_anu_0", "mean"),
        CAR_0_1=("CAR_anu_01", "mean"),
    )
    .reset_index()
)

for columna in ["Sorpresa_media", "DY_medio", "AR_0", "CAR_0_1"]:
    expectativas_cuartiles[f"{columna}_%"] = expectativas_cuartiles[columna] * 100

expectativas_cuartiles = expectativas_cuartiles[
    ["Q_sorpresa", "N", "Sorpresa_media_%", "DY_medio_%", "AR_0_%", "CAR_0_1_%"]
]


# ============================================================
# 21. EXPORTAR EXCEL COMPLETO
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

    # Ventana de fecha de pago
    pago_cuatro_pruebas.to_excel(writer, sheet_name="Pago_cuatro_pruebas", index=False)
    anuncio_cuatro_pruebas.to_excel(writer, sheet_name="Anuncio_cuatro_pruebas", index=False)
    pago_descomposicion.to_excel(writer, sheet_name="Pago_descomposicion", index=False)
    pago_escala_dividendo.to_excel(writer, sheet_name="Pago_escala_dividendo", index=False)
    pago_cortes.to_excel(writer, sheet_name="Pago_cortes", index=False)
    pago_cortes_nota.to_excel(writer, sheet_name="Pago_cortes_nota", index=False)
    ex_date_cuatro_pruebas.to_excel(writer, sheet_name="Ex_date_cuatro_pruebas", index=False)
    ex_date_ajuste.to_excel(writer, sheet_name="Ex_date_ajuste", index=False)

    # Modelo de expectativas
    forecast_cobertura.to_excel(writer, sheet_name="Forecast_cobertura", index=False)
    forecast_calidad.to_excel(writer, sheet_name="Forecast_calidad", index=False)
    expectativas_regresiones.to_excel(writer, sheet_name="Expectativas_regresiones", index=False)
    expectativas_signo.to_excel(writer, sheet_name="Expectativas_signo", index=False)
    expectativas_cuartiles.to_excel(writer, sheet_name="Expectativas_cuartiles", index=False)
    df_forecast.to_excel(writer, sheet_name="Base_633_con_forecast", index=False)


# ============================================================
# 22. MOSTRAR RESULTADOS CLAVE EN CONSOLA
# ============================================================

print("\nRESULTADO PRINCIPAL H1 - REGULAR CASH (N=633)")
print(
    tests_robustos_principales.loc[
        tests_robustos_principales["Variable"] == "AR_anuncio_+0",
        [
            "N", "Media_%", "Mediana_%", "%_positivos",
            "t_stat", "p_t", "p_wilcoxon", "p_sign",
            "p_bootstrap", "IC95_low_%", "IC95_high_%", "Pasa_4_pruebas",
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

print("\nVENTANA DE FECHA DE PAGO - CUATRO PRUEBAS")
print(
    pago_cuatro_pruebas[
        ["Variable", "N", "Media_%", "Mediana_%", "%_positivos",
         "p_t", "p_wilcoxon", "p_sign", "IC95_low_%", "IC95_high_%",
         "Pasa_4_pruebas"]
    ].round(4).to_string(index=False)
)

print("\nDESCOMPOSICION DEL REBOTE DEL DIA +1")
print(pago_descomposicion.round(4).to_string(index=False))

print("\nLA VENTANA DE PAGO CONTRA EL TAMANO DEL DIVIDENDO")
print(pago_escala_dividendo.round(4).to_string(index=False))

print("\nCORTES DE LA VENTANA DE PAGO")
print(pago_cortes.round(4).to_string(index=False))

print("\nEX DATE - CUATRO PRUEBAS")
print(
    ex_date_cuatro_pruebas[
        ["Variable", "N", "Media_%", "p_t", "p_wilcoxon", "p_sign",
         "IC95_low_%", "IC95_high_%", "Pasa_4_pruebas"]
    ].round(4).to_string(index=False)
)
print(ex_date_ajuste.round(4).to_string(index=False))

print("\nCOBERTURA DEL PRONOSTICO NAIVE")
print(forecast_cobertura.round(2).to_string(index=False))

print("\nCALIDAD DEL PRONOSTICO NAIVE")
print(forecast_calidad.round(6).to_string(index=False))

print("\nCAR CONTRA PARTE ESPERADA Y SORPRESA")
print(
    expectativas_regresiones[
        ["Modelo", "Errores", "N", "R2",
         "b_DY_esperado", "p_DY_esperado",
         "b_Sorpresa_DY", "p_Sorpresa_DY", "p_Wald"]
    ].round(4).to_string(index=False)
)

print("\nREACCION SEGUN EL SIGNO DE LA SORPRESA")
print(
    expectativas_signo[
        ["Muestra", "Grupo", "DY_medio_%", "Variable", "N", "Media_%",
         "p_t", "p_wilcoxon", "p_sign", "Pasa_4_pruebas"]
    ].round(4).to_string(index=False)
)

print("\nCUARTILES DE SORPRESA")
print(expectativas_cuartiles.round(4).to_string(index=False))

print(f"\nExcel creado en: {salida}")