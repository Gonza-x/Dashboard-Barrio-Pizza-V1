"""
forecasting.py
----------------
Proyección de consumo de la próxima semana, más inteligente que un promedio simple.

Con solo 6 semanas de histórico, un modelo complejo (ARIMA, Prophet, etc.) sobreajusta
y se vuelve una caja negra. En cambio usamos un método robusto y explicable:

  1. Detección de semanas atípicas con Median Absolute Deviation (MAD) — más
     resistente a outliers que un z-score con media/desvío estándar clásico.
  2. Ajuste de tendencia con Theil-Sen (mediana de todas las pendientes posibles
     entre pares de puntos) sobre las semanas "normales" — capta crecimiento o
     caída sostenida sin dejarse arrastrar por una sola semana rara.
  3. Backtesting simple: re-proyectamos la última semana conocida usando solo
     las anteriores, y medimos el error -> eso define la "confianza" que se
     muestra en el dashboard, en vez de mostrar un número falsamente preciso.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats


MAD_THRESHOLD = 3.5  # umbral estándar para "modified z-score" (Iglewicz & Hoaglin)


def modified_zscore_outliers(values: np.ndarray, threshold: float = MAD_THRESHOLD) -> np.ndarray:
    """Devuelve una máscara booleana: True donde el punto es atípico."""
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad == 0:
        # todos los valores casi iguales; usamos un fallback con desvío absoluto medio
        mad = np.mean(np.abs(values - median)) or 1e-9
    modified_z = 0.6745 * (values - median) / mad
    return np.abs(modified_z) > threshold


def _theil_sen_predict(weeks: np.ndarray, values: np.ndarray, target_week: int) -> tuple[float, float]:
    """Ajusta Theil-Sen y devuelve (proyección, pendiente)."""
    if len(weeks) < 2 or np.all(values == values[0]):
        # no hay suficiente variación / puntos -> usamos el último valor conocido
        return float(values[-1]), 0.0
    slope, intercept, _, _ = stats.theilslopes(values, weeks)
    pred = intercept + slope * target_week
    return float(pred), float(slope)


@dataclass
class Proyeccion:
    valor: float
    metodo: str
    tendencia: str  # "creciente" | "decreciente" | "estable"
    pendiente_semanal: float
    confianza: str  # "Alta" | "Media" | "Baja"
    error_backtest_pct: float | None
    semanas_excluidas: list[int] = field(default_factory=list)
    serie_original: list[float] = field(default_factory=list)


def proyectar_consumo(semanas: np.ndarray, valores: np.ndarray) -> Proyeccion:
    """
    semanas: array de números de semana (ej. [1,2,3,4,5,6])
    valores: consumo en unidad base para cada semana
    Devuelve la proyección para la semana siguiente (max(semanas)+1).
    """
    semanas = np.asarray(semanas, dtype=float)
    valores = np.asarray(valores, dtype=float)
    orden = np.argsort(semanas)
    semanas, valores = semanas[orden], valores[orden]
    target_week = semanas.max() + 1

    if len(valores) == 0:
        return Proyeccion(0.0, "sin_datos", "estable", 0.0, "Baja", None, [], [])

    if len(valores) < 3:
        # muy pocos datos para detectar outliers de forma confiable
        pred, slope = _theil_sen_predict(semanas, valores, target_week)
        pred = max(pred, 0.0)
        return Proyeccion(pred, "promedio_simple", "estable", slope, "Baja", None, [], valores.tolist())

    outlier_mask = modified_zscore_outliers(valores)
    semanas_excluidas = semanas[outlier_mask].astype(int).tolist()

    semanas_limpias = semanas[~outlier_mask]
    valores_limpios = valores[~outlier_mask]

    # Si excluir outliers nos deja con muy pocos puntos, no los excluimos
    if len(valores_limpios) < 3:
        semanas_limpias, valores_limpios = semanas, valores
        semanas_excluidas = []

    pred, slope = _theil_sen_predict(semanas_limpias, valores_limpios, target_week)
    pred = max(pred, 0.0)

    media = float(np.mean(valores_limpios))
    if media > 0 and abs(slope) / media > 0.03:
        tendencia = "creciente" if slope > 0 else "decreciente"
    else:
        tendencia = "estable"

    # --- Backtest: reentrenar con todas menos la última semana limpia, predecir esa última ---
    error_pct = None
    if len(valores_limpios) >= 4:
        s_train, v_train = semanas_limpias[:-1], valores_limpios[:-1]
        s_test, v_test = semanas_limpias[-1], valores_limpios[-1]
        pred_test, _ = _theil_sen_predict(s_train, v_train, s_test)
        if v_test != 0:
            error_pct = abs(pred_test - v_test) / v_test * 100
        else:
            error_pct = None

    if error_pct is None:
        confianza = "Media" if len(valores_limpios) >= 4 else "Baja"
    elif error_pct < 10:
        confianza = "Alta"
    elif error_pct < 25:
        confianza = "Media"
    else:
        confianza = "Baja"

    return Proyeccion(
        valor=pred,
        metodo="theil_sen_robusto",
        tendencia=tendencia,
        pendiente_semanal=slope,
        confianza=confianza,
        error_backtest_pct=error_pct,
        semanas_excluidas=semanas_excluidas,
        serie_original=valores.tolist(),
    )
