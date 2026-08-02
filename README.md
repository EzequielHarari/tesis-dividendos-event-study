# Tesis — La reacción del mercado frente a los anuncios de dividendos

Código de la tesis de Licenciatura en Finanzas de la **Universidad de San Andrés (UdeSA)**:

> **La reacción del mercado frente a los anuncios de dividendos: evidencia del mercado argentino mediante un _Event Study_**
> Ezequiel Mario Harari y María Candelaria Seleme Durán — 2026

## Contenido

El archivo `codigo_tesis.py` reproduce todo el análisis empírico del trabajo:

- Cálculo de retornos anormales (AR) y acumulados (CAR) por el método _market-adjusted_, usando el S&P Merval como _benchmark_, para las fechas de anuncio y de pago.
- Estadística descriptiva y pruebas de significatividad **paramétricas** (test _t_ de una muestra).
- Pruebas **no paramétricas** (Wilcoxon _signed-rank_, _sign test_) y procedimientos **_bootstrap_** (10.000 remuestreos, IC al 95%).
- Segmentación de los dividendos ordinarios por _Dividend Yield_ en terciles, cuartiles y quintiles.
- Modelos de **regresión**: cuartiles, _Dividend Yield_ continuo, transformación logarítmica, efectos fijos por año y errores estándar robustos (HC3).
- Pruebas de **robustez**: exclusión de observaciones extremas (percentiles 1 y 99) y _winsorización_ del _Dividend Yield_.

## Requisitos

```
python >= 3.10
pandas
numpy
scipy
statsmodels
openpyxl
```

```bash
pip install pandas numpy scipy statsmodels openpyxl
```

## Datos

Los precios provienen de Bloomberg (serie `PX_LAST`) y **no se incluyen** en este repositorio por restricciones de licencia. Las rutas a los archivos de entrada y de salida deben ajustarse al inicio y al final del script.

## Cómo ejecutar

```bash
python codigo_tesis.py
```

El script genera un archivo Excel con todas las tablas de resultados y de robustez utilizadas en el trabajo.
