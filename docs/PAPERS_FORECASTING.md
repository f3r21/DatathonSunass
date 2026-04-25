# Papers y referencias — Fundamentación del pipeline de forecasting

> Base bibliográfica para el deck del Datathon SUNASS 2026. Cada elección
> metodológica en `src/modeling/forecasting.py` y `src/modeling/anomalias.py`
> está respaldada por los papers listados aquí, con una línea de justificación.

## 1. Baselines estadísticos (ETS, SARIMA, Holt-Winters)

**Hyndman, R. J., & Athanasopoulos, G. (2021).** *Forecasting: Principles and
Practice* (3.ª ed.). OTexts, Melbourne. https://otexts.com/fpp3/

Manual canónico de forecasting. Define el marco ETS, la taxonomía SARIMA y las
métricas de evaluación (MAPE, sMAPE, MASE). Lo usa el mentor del Datathon en
sus scripts R (`forecast::auto.arima`, `forecast::ets`, `forecast::hw`).

**Hyndman, R. J., Koehler, A. B., Snyder, R. D., & Grose, S. (2002).** "A state
space framework for automatic forecasting using exponential smoothing methods".
*International Journal of Forecasting*, 18(3), 439–454.

Base teórica del ETS en espacio de estados. Justifica la selección automática
entre modelos A/M, aditivo/multiplicativo, con o sin estacionalidad.

**Winters, P. R. (1960).** "Forecasting sales by exponentially weighted moving
averages". *Management Science*, 6(3), 324–342.

Holt-Winters multiplicativo — el clásico para series con varianza que crece
con la media.

**Hyndman, R. J., & Koehler, A. B. (2006).** "Another look at measures of
forecast accuracy". *International Journal of Forecasting*, 22(4), 679–688.

Justifica usar sMAPE sobre MAPE cuando los valores observados pueden ser muy
pequeños o cero (caso interrupciones: meses con pocos eventos).

**Tashman, L. J. (2000).** "Out-of-sample tests of forecasting accuracy: an
analysis and review". *International Journal of Forecasting*, 16(4), 437–450.

Metodología de backtest *rolling-origin* (`walk_forward_backtest`).

---

## 2. ML con features engineered (LightGBM + lags)

**Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2022).** "The M5
accuracy competition: Results, findings, and conclusions". *International
Journal of Forecasting*, 38(4), 1346–1364.

M5 fue el competitor más influyente reciente (42 840 series de Walmart). Los
top-5 fueron todos **LightGBM con lags y features calendáricos**, no deep
learning. Esto justifica nuestro `forecast_lgbm_lags`.

**Januschowski, T., Gasthaus, J., Wang, Y., Salinas, D., Flunkert, V.,
Bohlke-Schneider, M., & Callot, L. (2020).** "Criteria for classifying
forecasting methods". *International Journal of Forecasting*, 36(1), 167–177.

Define el eje global–local (un modelo para todas las series vs uno por serie)
y el eje estadístico–ML. Útil para defender la elección de modelos *locales*
cuando las series SUNASS son heterogéneas por región.

**Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2018).** "Statistical
and Machine Learning forecasting methods: Concerns and ways forward". *PLoS
ONE*, 13(3), e0194889.

Evidencia empírica de que los métodos estadísticos clásicos (ETS/ARIMA) siguen
siendo competitivos frente a ML puro, especialmente con poca data. Respalda el
pipeline mixto estadístico + ML.

---

## 3. Ensembles y combinación de pronósticos

**Bates, J. M., & Granger, C. W. J. (1969).** "The combination of forecasts".
*Operational Research Quarterly*, 20(4), 451–468.

Teorema fundacional de combinación de pronósticos: el ensemble reduce el MSE
cuando los componentes tienen errores no perfectamente correlacionados.

**Clemen, R. T. (1989).** "Combining forecasts: A review and annotated
bibliography". *International Journal of Forecasting*, 5(4), 559–583.

Hallazgo clásico: **el promedio simple suele igualar o superar a combinaciones
ponderadas optimizadas** cuando la muestra es corta, porque estimar pesos
introduce varianza. Justifica `forecast_ensemble(weights=None)`.

**Petropoulos, F., et al. (2022).** "Forecasting: theory and practice".
*International Journal of Forecasting*, 38(3), 705–871.

Revisión enciclopédica del estado del arte con 73 coautores. Referencia
general para situar nuestras decisiones en el contexto actual.

---

## 4. Detección de anomalías multivariada

**Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008).** "Isolation Forest". En
*Proceedings of the 8th IEEE International Conference on Data Mining*, 413–422.

Paper original del Isolation Forest. `isolation_forest_scan` lo usa sobre
`(ph, cloro, temperatura)` para flaggear lecturas multivariadamente raras.

**Breunig, M. M., Kriegel, H.-P., Ng, R. T., & Sander, J. (2000).** "LOF:
identifying density-based local outliers". *ACM SIGMOD Record*, 29(2), 93–104.

Local Outlier Factor — útil cuando las anomalías son locales al clúster
geográfico/temporal de cada estación.

**Truong, C., Oudre, L., & Vayatis, N. (2020).** "Selective review of offline
change point detection methods". *Signal Processing*, 167, 107299.

Revisión que justifica el uso de **PELT** (Pruned Exact Linear Time) en
`ruptures_changepoints` para detectar cambios de régimen sostenidos en series
de sensores MOREA.

**Killick, R., Fearnhead, P., & Eckley, I. A. (2012).** "Optimal detection of
changepoints with a linear computational cost". *Journal of the American
Statistical Association*, 107(500), 1590–1598.

Paper original del algoritmo PELT que usa `ruptures`.

---

## 5. Deep learning para series — siguientes pasos, **no** en la entrega

**Zeng, A., Chen, M., Zhang, L., & Xu, Q. (2023).** "Are transformers effective
for time series forecasting?" *Proceedings of the AAAI Conference*, 37(9),
11121–11128.

**Critique paper clave.** Muestra que una regresión lineal simple (DLinear)
iguala o supera a Informer/Autoformer/FEDformer en LTSF con datasets realistas
del benchmark. Justifica por qué **no** entrenamos transformers de cero en
48 h con series cortas.

**Oreshkin, B. N., Carpov, D., Chapados, N., & Bengio, Y. (2020).** "N-BEATS:
Neural basis expansion analysis for interpretable time series forecasting".
*ICLR 2020*.

Si hubiera tiempo, este sería el primer candidato DL por su interpretabilidad
(stacks trend + seasonality + residual) y por ganar la M4 cuando se combinó
con ES-RNN.

**Zhou, H., Zhang, S., Peng, J., et al. (2021).** "Informer: Beyond efficient
transformer for long sequence time-series forecasting". *AAAI 2021* (Best
Paper). Repo: https://github.com/zhouhaoyi/Informer2020

**Wu, H., Xu, J., Wang, J., & Long, M. (2021).** "Autoformer: Decomposition
transformers with auto-correlation for long-term series forecasting".
*NeurIPS 2021*. Repo: https://github.com/thuml/Autoformer

**Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023).** "A time
series is worth 64 words: Long-term forecasting with transformers" (PatchTST).
*ICLR 2023*.

**Challu, C., Olivares, K. G., Oreshkin, B. N., et al. (2023).** "N-HiTS:
Neural hierarchical interpolation for time series forecasting". *AAAI 2023*.

**Tuli, S., Casale, G., & Jennings, N. R. (2022).** "TranAD: Deep
transformer networks for anomaly detection in multivariate time series data".
*VLDB 2022*. Repo: https://github.com/imperial-qore/TranAD

Transformer específico para **anomalías** multivariadas — candidato para slide
de "Siguientes pasos" dada la naturaleza IoT/SCADA de MOREA.

---

## 6. Evaluación probabilística (si se reporta intervalo de confianza)

**Gneiting, T., & Raftery, A. E. (2007).** "Strictly proper scoring rules,
prediction, and estimation". *Journal of the American Statistical
Association*, 102(477), 359–378.

Define CRPS (Continuous Ranked Probability Score) como la métrica correcta
para forecasts probabilísticos. Referencia obligatoria si la solución incluye
intervalos de predicción.

---

## 7. Clase desbalanceada (Plantilla A, no forecasting pero del mismo paper trail)

**Saito, T., & Rehmsmeier, M. (2015).** "The precision-recall plot is more
informative than the ROC plot when evaluating binary classifiers on imbalanced
datasets". *PLoS ONE*, 10(3), e0118432.

Justifica por qué `ModelEvaluation` reporta PR-AUC y no solo ROC-AUC cuando
la prevalencia de `evento_critico` es ~2–5%.

**Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002).**
"SMOTE: Synthetic Minority Over-sampling Technique". *Journal of Artificial
Intelligence Research*, 16, 321–357.

Alternativa de oversampling si `class_weight='balanced'` no alcanza.
Disponible via `imbalanced-learn` (ya en `pyproject.toml`).

---

## Cómo citar en el deck

Cada slide que introduzca un método debe llevar una línea al pie con la cita
en formato corto, p.e.:

> *LightGBM + lags (M5 winner pattern, Makridakis et al. 2022)*
> *SARIMA(1,1,1)(1,1,1,12) — Hyndman & Athanasopoulos (2021)*
> *DLinear beats transformers on short series — Zeng et al. (AAAI 2023)*

Slide de "Referencias" al final del deck: volcar la bibliografía completa de
este documento.
