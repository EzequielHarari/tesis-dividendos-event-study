# Tesis — La reacción del mercado frente a los anuncios de dividendos

Código de la tesis de Licenciatura en Finanzas de la **Universidad de San Andrés (UdeSA)**:

> **La reacción del mercado frente a los anuncios de dividendos: evidencia del mercado argentino mediante un _Event Study_**
> Ezequiel Mario Harari y María Candelaria Seleme Durán — 2026

## Contenido

El archivo `codigo_tesis_final_github.py` reproduce todo el análisis empírico del trabajo:

- Cálculo de retornos anormales (AR) y acumulados (CAR) por el método _market-adjusted_, usando el S&P Merval como _benchmark_, para las fechas de anuncio y de pago.
- Estadística descriptiva y pruebas de significatividad **paramétricas** (test _t_ de una muestra).
- Pruebas **no paramétricas** (Wilcoxon _signed-rank_, _sign test_) y procedimientos **_bootstrap_** (10.000 remuestreos, IC al 95%), bajo un criterio conjunto explícito: una diferencia se reporta como significativa cuando las cuatro pruebas coinciden.
- Segmentación de los dividendos ordinarios por _Dividend Yield_ en terciles, cuartiles y quintiles.
- Modelos de **regresión**: cuartiles, _Dividend Yield_ continuo, transformación logarítmica, efectos fijos por año, errores estándar robustos (HC3) y errores clusterizados por empresa.
- Pruebas de **robustez**: exclusión de observaciones extremas (percentiles 1 y 99) y _winsorización_ del _Dividend Yield_.

Las secciones 19 y 20 del script incorporan dos bloques adicionales:

- **Ventana de fecha de pago.** Las cuatro pruebas día por día, la descomposición del rebote del día +1 en reversión y _drift_ con la ventana de anuncio como placebo, la relación entre el patrón y el tamaño del dividendo, y cortes por magnitud del _yield_, subperíodo y liquidez. Se incluye la _ex date_ como respaldo, con su _drop-off ratio_.
- **Modelo naive de expectativas.** Pronostica el dividendo del anuncio con el último monto anunciado dentro de los 365 días previos; si la empresa no pagó en ese lapso, el pronóstico es cero. Evalúa la calidad del pronóstico, descompone el CAR en parte esperada y sorpresa, y contrasta mediante un test de **Wald** si esa descomposición agrega algo sobre la regresión del CAR contra el _Dividend Yield_. Se reporta en unidades de _yield_ y, como contraste, en dividendo nominal.

La sección 18 verifica los resultados centrales del trabajo —el AR medio del día del anuncio y el coeficiente del _Dividend Yield_ continuo— contra los valores reportados, y los exporta en la hoja `Chequeos_PTG`.

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

Los precios provienen de Bloomberg (serie `PX_LAST`) y **no se incluyen** en este repositorio por restricciones de licencia. El script espera un archivo Excel con la hoja `Dividendos_val`. Las rutas de entrada y de salida se configuran al inicio del script, en las variables `archivo` y `salida`.

## Cómo ejecutar

```bash
python codigo_tesis_final_github.py
```

El script genera `resultados_tesis_completo.xlsx`, con 49 hojas: las bases de 774 y 633 eventos, los descriptivos y las pruebas de significatividad, las tablas por tipo de dividendo y por grupo de _Dividend Yield_, las tablas auxiliares de presentación, las regresiones y sus pruebas de robustez, y las hojas de los dos bloques adicionales.
