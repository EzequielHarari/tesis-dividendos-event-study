"""
================================================================================
Addenda de la tesis de Licenciatura en Finanzas (Universidad de San Andres)

    "La reaccion del mercado frente a los anuncios de dividendos:
     evidencia del mercado argentino mediante un Event Study"

Autores : Ezequiel Mario Harari y Maria Candelaria Seleme Duran
Ano     : 2026

Complemento de codigo_tesis.py. Reutiliza sus mismas funciones de construccion
de la base (crear_AR, crear_CAR_rango, convertir_fecha_excel_mixta) y su misma
bateria de pruebas (test t, Wilcoxon, sign test y bootstrap), y agrega los dos
bloques incorporados para la defensa final:

  A. Ventana de fecha de pago
     Documenta la dinamica de caida y rebote alrededor de la fecha de pago:
     las cuatro pruebas dia por dia, la descomposicion del rebote en reversion
     y drift, la relacion con el tamano del dividendo y los cortes por
     magnitud, subperiodo y liquidez. Se agrega la ex date como respaldo.

  B. Modelo naive de expectativas
     Pronostica el dividendo del anuncio con el ultimo monto anunciado dentro
     de los 365 dias previos; si la empresa no pago en ese lapso, el
     pronostico es cero. Descompone el CAR en la parte esperada y la sorpresa
     y contrasta, mediante un test de Wald, si esa descomposicion agrega algo
     sobre la regresion del CAR contra el Dividend Yield.

Ninguno de los dos bloques altera los resultados reportados en el trabajo: se
verifican contra ellos en la seccion 8.

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
import statsmodels.formula.api as smf


# ============================================================
# 0. CONFIGURACION
# ============================================================

archivo = "/Users/Usuario/Documents/Tesis/Dividendos_datos_new.xlsx"
hoja_base = "Dividendos_val"

salida = "/Users/Usuario/Documents/Tesis/addenda_defensa.xlsx"

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
# 2. CONSTRUCCION DE LA BASE (identica a codigo_tesis.py)
# ============================================================

def crear_AR(df, evento="anuncio", ventana=2):
    """Retorno anormal diario: retorno de la accion menos retorno del indice."""
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


def _fmt_dia(dia):
    return "0" if dia == 0 else f"{dia:+d}"


def crear_CAR_rango(df, evento="anuncio", inicio=-1, fin=1):
    """Retorno anormal acumulado entre dos dias de la ventana."""
    df = df.copy()

    columnas_ar = [
        f"AR_{evento}_{dia:+d}"
        for dia in range(inicio, fin + 1)
    ]

    nombre_car = f"CAR_{evento}_[{_fmt_dia(inicio)},{_fmt_dia(fin)}]"
    df[nombre_car] = df[columnas_ar].sum(axis=1, skipna=False)

    return df


def crear_AR_ex_date(df):
    """AR alrededor de la ex date.

    Se construye con las mismas columnas de retorno de la base que usan los AR
    de anuncio y de pago. La ex date no se reporta en el cuerpo del trabajo;
    se incluye aqui como respaldo de la lectura de la ventana de pago.
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


def preparar_datos(df):
    """Agrega los nombres simples que necesitan las formulas de statsmodels.

    Los nombres originales de las columnas llevan signos y corchetes, que
    patsy interpreta como operadores. Las columnas originales se conservan.
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

    datos["DY"] = pd.to_numeric(datos["Dividend Yield anuncio"], errors="coerce")
    datos["DPA"] = pd.to_numeric(datos["Ammount"], errors="coerce")
    datos["P0"] = pd.to_numeric(datos["Precio_fecha de aviso"], errors="coerce")

    datos["Fecha_anuncio"] = datos["Fecha de aviso"].map(convertir_fecha_excel_mixta)
    datos["Anio"] = datos["Fecha_anuncio"].dt.year

    return datos


# ============================================================
# 3. TESTS ROBUSTOS: T, WILCOXON, SIGN Y BOOTSTRAP
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
    """Las cuatro pruebas sobre una serie, mas el criterio conjunto.

    Pasa_4_pruebas vale 1 cuando el test t, el Wilcoxon y el sign test son
    significativos al 5% y el intervalo bootstrap excluye el cero. Es el
    criterio de lectura declarado en el trabajo: si las cuatro coinciden, la
    conclusion no depende del supuesto estadistico elegido.
    """
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

    # 5. Criterio conjunto
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


def crear_tests_robustos_variables(df, variables, etiquetas=None):
    """Bateria de cuatro pruebas sobre una lista de columnas."""
    resultados = []

    for i, var in enumerate(variables):
        fila = {"Variable": etiquetas[i] if etiquetas else var}
        fila.update(_tests_una_serie(df[var]))
        resultados.append(fila)

    return pd.DataFrame(resultados)


# ============================================================
# 4. FUNCIONES PARA REGRESIONES
# ============================================================

def ajustar_ols(formula, data, hc3=True, cluster=None):
    """MCO con matriz robusta HC3 o con errores clusterizados.

    cluster: nombre de la columna de agrupamiento, por ejemplo "Company".
    La muestra tiene 633 eventos de 43 empresas, de modo que las
    observaciones de una misma empresa no son independientes entre si. El
    clustering relaja ese supuesto: modifica los errores estandar, no los
    coeficientes.
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


def fila_modelo(modelo, nombre_modelo, variables, wald=None):
    """Resumen compacto de un modelo.

    Devuelve intercepto, las variables pedidas con su error estandar y su
    p-valor y, de manera opcional, el p-valor de una restriccion lineal.
    """
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


# ============================================================
# 5. BLOQUE A: VENTANA DE FECHA DE PAGO
# ============================================================

def crear_descomposicion_reversion(datos):
    """Separa el rebote del dia +1 en reversion y drift.

    El intercepto mide la parte del AR(+1) que no se explica por la caida del
    dia anterior; el coeficiente negativo mide la reversion. La ventana de
    anuncio se estima como placebo: si alli el intercepto no es significativo,
    el patron es propio de la fecha de pago y no una regularidad general de
    estas acciones.
    """
    m_pago = ajustar_ols("AR_pago_1 ~ AR_pago_0", datos)
    m_anuncio = ajustar_ols("AR_anu_1 ~ AR_anu_0", datos)

    tabla = pd.DataFrame([
        fila_modelo(m_pago, "Pago: AR(+1) ~ AR(0)", ["AR_pago_0"]),
        fila_modelo(m_anuncio, "Anuncio (placebo): AR(+1) ~ AR(0)", ["AR_anu_0"]),
    ])

    tabla["Nota"] = [
        "Intercepto = drift no explicado por reversion; coeficiente < 0 = reversion",
        "Placebo: intercepto no significativo implica patron propio del pago",
    ]

    return tabla


def crear_escala_dividendo(datos):
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
        filas.append(
            fila_modelo(ajustar_ols(f"{variable} ~ DY", datos), etiqueta, ["DY"])
        )

    return pd.DataFrame(filas)


def crear_cortes_pago(datos, columna_grupo, etiqueta):
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


# ============================================================
# 6. BLOQUE B: MODELO NAIVE DE EXPECTATIVAS
# ============================================================

def construir_pronostico_naive(df, dias=DIAS_VENTANA_FORECAST):
    """Pronostico ingenuo del dividendo del anuncio.

    Regla: si la empresa anuncio otro dividendo ordinario dentro de los dias
    previos indicados, se espera el mismo monto por accion; si no pago en ese
    lapso, el pronostico es cero. Se construyen dos versiones de la sorpresa:

      - en yield, con el ultimo monto valuado al precio del anuncio, que es la
        unidad comparable a lo largo de treinta anos de inflacion;
      - en dividendo nominal, que se reporta solo como contraste.
    """
    datos = df.sort_values(["Company", "Fecha_anuncio"]).reset_index(drop=True)

    dpa_previo, n_previos, dias_previos = [], [], []

    for _, fila in datos.iterrows():
        fecha = fila["Fecha_anuncio"]

        previos = datos.loc[
            (datos["Company"] == fila["Company"])
            & (datos["Fecha_anuncio"] < fecha)
            & (datos["Fecha_anuncio"] >= fecha - pd.Timedelta(days=dias))
        ]

        n_previos.append(len(previos))

        if len(previos) == 0:
            dpa_previo.append(0.0)
            dias_previos.append(np.nan)
        else:
            ultimo = previos.sort_values("Fecha_anuncio").iloc[-1]
            dpa_previo.append(float(ultimo["DPA"]))
            dias_previos.append((fecha - ultimo["Fecha_anuncio"]).days)

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

    modelo = ajustar_ols(formula, datos, cluster=cluster)

    variables = [esperado, sorpresa] + ([extra] if extra else [])
    fila = fila_modelo(modelo, etiqueta, variables, wald=f"{esperado} = {sorpresa}")
    fila["Variable_dependiente"] = y
    fila["Errores"] = "Cluster por empresa" if cluster else "HC3"

    return fila


def clasificar_signo_sorpresa(valor, tolerancia=0.001):
    """Clasifica el anuncio segun supere, iguale o no alcance el pronostico."""
    if pd.isna(valor):
        return np.nan
    if valor > tolerancia:
        return "Sorpresa positiva"
    if valor < -tolerancia:
        return "Sorpresa negativa"
    return "Sin sorpresa (+-0,1pp)"


# ============================================================
# 7. EJECUTAR TODO
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

# AR de ex date, que codigo_tesis.py no construye
df_final = crear_AR_ex_date(df_final)

# ------------------------------------------------------------
# MUESTRA PRINCIPAL: 633 REGULAR CASH
# ------------------------------------------------------------
df_regular = preparar_datos(df_final.loc[df_final["Type"].eq("Regular Cash")].copy())

print("============================================================")
print(f"Muestra total                 : {len(df_final)}")
print(f"Muestra principal Regular Cash: {len(df_regular)}")
print("============================================================")

if len(df_final) != 774:
    print("ADVERTENCIA: la muestra total no tiene 774 eventos.")
if len(df_regular) != 633:
    print("ADVERTENCIA: Regular Cash no tiene 633 eventos.")


# ------------------------------------------------------------
# A1. LA VENTANA DE PAGO CON LAS CUATRO PRUEBAS
# ------------------------------------------------------------

pago_cuatro_pruebas = crear_tests_robustos_variables(
    df_regular,
    ["AR_pago_m1", "AR_pago_0", "AR_pago_1", "AR_pago_2",
     "CAR_pago_01", "CAR_pago_02", "CAR_pago_m11"],
    ["AR pago (-1)", "AR pago (0)", "AR pago (+1)", "AR pago (+2)",
     "CAR pago [0,+1]", "CAR pago [0,+2]", "CAR pago [-1,+1]"],
)

anuncio_cuatro_pruebas = crear_tests_robustos_variables(
    df_regular,
    ["AR_anu_0", "AR_anu_1", "CAR_anu_01"],
    ["AR anuncio (0)", "AR anuncio (+1)", "CAR anuncio [0,+1]"],
)


# ------------------------------------------------------------
# A2. DESCOMPOSICION DEL REBOTE Y ESCALA CON EL DIVIDENDO
# ------------------------------------------------------------

pago_descomposicion = crear_descomposicion_reversion(df_regular)
pago_escala_dividendo = crear_escala_dividendo(df_regular)


# ------------------------------------------------------------
# A3. CORTES POR MAGNITUD, SUBPERIODO Y LIQUIDEZ
# ------------------------------------------------------------

df_regular["Q_DY"] = pd.qcut(
    df_regular["DY"], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop"
)
df_regular["Subperiodo"] = pd.cut(
    df_regular["Anio"],
    bins=[1995, 2009, 2017, 2026],
    labels=["1996-2009", "2010-2017", "2018-2026"],
)

empresas_frecuentes = df_regular["Company"].value_counts().head(6).index.tolist()
df_regular["Tamano"] = np.where(
    df_regular["Company"].isin(empresas_frecuentes),
    "6 con mas eventos",
    "Resto",
)

pago_cortes = pd.concat([
    crear_cortes_pago(df_regular, "Q_DY", "Cuartil de Dividend Yield"),
    crear_cortes_pago(df_regular, "Subperiodo", "Subperiodo"),
    crear_cortes_pago(df_regular, "Tamano", "Tamano / liquidez"),
], ignore_index=True)

pago_cortes_nota = pd.DataFrame([{
    "Empresas_mas_frecuentes": ", ".join(empresas_frecuentes),
    "Criterio": "seis empresas con mas anuncios dentro de la muestra Regular Cash",
}])


# ------------------------------------------------------------
# A4. EX DATE (RESPALDO)
# ------------------------------------------------------------

ex_date_cuatro_pruebas = crear_tests_robustos_variables(
    df_regular,
    ["AR_ex_m1", "AR_ex_0", "AR_ex_1"],
    ["AR ex date (-1)", "AR ex date (0)", "AR ex date (+1)"],
)

ex_date_ajuste = pd.DataFrame([
    fila_modelo(
        ajustar_ols("AR_ex_0 ~ DY", df_regular),
        "AR ex (0) ~ DY  [drop-off ratio]",
        ["DY"],
    ),
    fila_modelo(
        ajustar_ols("AR_ex_m1 ~ DY", df_regular),
        "AR ex (-1) ~ DY",
        ["DY"],
    ),
])


# ------------------------------------------------------------
# B0. CONSTRUCCION DEL PRONOSTICO Y COBERTURA
# ------------------------------------------------------------

df_forecast = construir_pronostico_naive(df_regular)
df_forecast_con_historia = df_forecast.loc[df_forecast["Sin_pago_previo"] == 0].copy()

forecast_cobertura = pd.DataFrame([{
    "N_total": len(df_forecast),
    "N_con_pago_previo_12m": int((df_forecast["Sin_pago_previo"] == 0).sum()),
    "N_sin_pago_previo_12m": int((df_forecast["Sin_pago_previo"] == 1).sum()),
    "Dias_medios_al_pago_previo": np.nanmean(df_forecast["Dias_desde_previo"]),
    "Ventana_dias": DIAS_VENTANA_FORECAST,
}])


# ------------------------------------------------------------
# B1. CALIDAD DEL PRONOSTICO
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# B2. CAR CONTRA PARTE ESPERADA Y SORPRESA
# ------------------------------------------------------------

filas_expectativas = []

# Referencia: la regresion del CAR contra el yield total, como en el trabajo.
filas_expectativas.extend([
    fila_modelo(
        ajustar_ols("CAR_anu_01 ~ DY", df_forecast),
        "REFERENCIA: CAR[0,+1] ~ DY", ["DY"],
    ),
    fila_modelo(
        ajustar_ols("CAR_anu_02 ~ DY", df_forecast),
        "REFERENCIA: CAR[0,+2] ~ DY", ["DY"],
    ),
    fila_modelo(
        ajustar_ols("CAR_anu_01 ~ DY", df_forecast, cluster="Company"),
        "REFERENCIA: CAR[0,+1] ~ DY, cluster por empresa", ["DY"],
    ),
])

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
        "Yield: CAR[0,+1], con efectos fijos por ano", extra="C(Anio)",
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


# ------------------------------------------------------------
# B3. REACCION SEGUN EL SIGNO DE LA SORPRESA
# ------------------------------------------------------------

filas_signo = []

for muestra, etiqueta in [
    (df_forecast, "Toda la muestra"),
    (df_forecast_con_historia, "Solo con pago previo"),
]:
    aux = muestra.copy()
    aux["Signo_sorpresa"] = aux["Sorpresa_DY"].map(clasificar_signo_sorpresa)

    for grupo, sub in aux.groupby("Signo_sorpresa", observed=False):
        for variable, nombre in [("AR_anu_0", "AR(0)"), ("CAR_anu_01", "CAR[0,+1]")]:
            fila = {"Muestra": etiqueta, "Grupo": grupo, "Variable": nombre}
            fila.update(_tests_una_serie(sub[variable]))
            filas_signo.append(fila)

expectativas_signo = pd.DataFrame(filas_signo)

df_forecast["Q_sorpresa"] = pd.qcut(
    df_forecast["Sorpresa_DY"], 4, labels=["S1", "S2", "S3", "S4"], duplicates="drop"
)

expectativas_cuartiles = (
    df_forecast.groupby("Q_sorpresa", observed=False)
    .agg(
        N=("CAR_anu_01", "size"),
        Sorpresa_media=("Sorpresa_DY", "mean"),
        AR_0=("AR_anu_0", "mean"),
        CAR_0_1=("CAR_anu_01", "mean"),
    )
    .reset_index()
)

for columna in ["Sorpresa_media", "AR_0", "CAR_0_1"]:
    expectativas_cuartiles[f"{columna}_%"] = expectativas_cuartiles[columna] * 100

expectativas_cuartiles = expectativas_cuartiles[
    ["Q_sorpresa", "N", "Sorpresa_media_%", "AR_0_%", "CAR_0_1_%"]
]


# ============================================================
# 8. CHEQUEOS CONTRA RESULTADOS CENTRALES DEL TRABAJO
# ============================================================

chequeos = []

h1 = _tests_una_serie(df_regular["AR_anu_0"])
chequeos.append({
    "Chequeo": "H1_AR0_633",
    "Valor": h1["Media_%"],
    "Esperado_aprox": 0.413,
    "Detalle": f"p_t={h1['p_t']:.8f}; p_w={h1['p_wilcoxon']:.8f}; p_sign={h1['p_sign']:.8f}",
})

referencia = ajustar_ols("CAR_anu_01 ~ DY", df_regular)
chequeos.append({
    "Chequeo": "DY_continuo_0_1",
    "Valor": referencia.params["DY"],
    "Esperado_aprox": 0.2349,
    "Detalle": f"p={referencia.pvalues['DY']:.6f}",
})

chequeos_addenda = pd.DataFrame(chequeos)


# ============================================================
# 9. EXPORTAR EXCEL
# ============================================================

with pd.ExcelWriter(salida, engine="openpyxl") as writer:
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

    # Chequeos y base
    chequeos_addenda.to_excel(writer, sheet_name="Chequeos_addenda", index=False)
    df_forecast.to_excel(writer, sheet_name="Base_633_con_forecast", index=False)


# ============================================================
# 10. MOSTRAR RESULTADOS CLAVE EN CONSOLA
# ============================================================

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)

print("\nA1. VENTANA DE FECHA DE PAGO - CUATRO PRUEBAS")
print(
    pago_cuatro_pruebas[
        ["Variable", "N", "Media_%", "Mediana_%", "%_positivos",
         "p_t", "p_wilcoxon", "p_sign", "IC95_low_%", "IC95_high_%",
         "Pasa_4_pruebas"]
    ].round(4).to_string(index=False)
)

print("\nA1. VENTANA DE ANUNCIO - COMPARACION")
print(
    anuncio_cuatro_pruebas[
        ["Variable", "N", "Media_%", "p_t", "p_wilcoxon", "p_sign",
         "Pasa_4_pruebas"]
    ].round(4).to_string(index=False)
)

print("\nA2. DESCOMPOSICION DEL REBOTE DEL DIA +1")
print(pago_descomposicion.round(4).to_string(index=False))

print("\nA2. LA VENTANA DE PAGO CONTRA EL TAMANO DEL DIVIDENDO")
print(pago_escala_dividendo.round(4).to_string(index=False))

print("\nA3. CORTES")
print(pago_cortes.round(4).to_string(index=False))

print("\nA4. EX DATE")
print(
    ex_date_cuatro_pruebas[
        ["Variable", "N", "Media_%", "p_t", "p_wilcoxon", "p_sign",
         "IC95_low_%", "IC95_high_%", "Pasa_4_pruebas"]
    ].round(4).to_string(index=False)
)
print(ex_date_ajuste.round(4).to_string(index=False))

print("\nB0. COBERTURA DEL PRONOSTICO NAIVE")
print(forecast_cobertura.round(2).to_string(index=False))

print("\nB1. CALIDAD DEL PRONOSTICO NAIVE")
print(forecast_calidad.round(6).to_string(index=False))

print("\nB2. CAR CONTRA PARTE ESPERADA Y SORPRESA (version en yield)")
print(
    expectativas_regresiones.loc[
        ~expectativas_regresiones["Modelo"].str.startswith("Nominal"),
        ["Modelo", "Errores", "N", "R2",
         "b_DY_esperado", "p_DY_esperado",
         "b_Sorpresa_DY", "p_Sorpresa_DY", "p_Wald"],
    ].round(4).to_string(index=False)
)

print("\nB2. VERSION EN DIVIDENDO NOMINAL (contraste)")
print(
    expectativas_regresiones.loc[
        expectativas_regresiones["Modelo"].str.startswith("Nominal"),
        ["Modelo", "N", "R2",
         "b_DPA_esperado", "p_DPA_esperado",
         "b_Sorpresa_DPA", "p_Sorpresa_DPA", "p_Wald"],
    ].round(6).to_string(index=False)
)

print("\nB3. REACCION SEGUN EL SIGNO DE LA SORPRESA")
print(
    expectativas_signo[
        ["Muestra", "Grupo", "Variable", "N", "Media_%",
         "p_t", "p_wilcoxon", "p_sign", "Pasa_4_pruebas"]
    ].round(4).to_string(index=False)
)

print("\nB3. CUARTILES DE SORPRESA")
print(expectativas_cuartiles.round(4).to_string(index=False))

print("\nCHEQUEOS CONTRA EL TRABAJO")
print(chequeos_addenda.to_string(index=False))

print(f"\nExcel creado en: {salida}")