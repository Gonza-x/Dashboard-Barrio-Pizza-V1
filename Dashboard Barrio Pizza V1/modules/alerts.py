"""
alerts.py
----------
Motor de alertas. Para cada (sucursal, ingrediente):

  necesidad_real = consumo_proyectado - inventario_actual
  necesidad_formatos = ceil(necesidad_real / tamaño_formato)   # no existe medio saco
  diff = pedido_actual (en unidad base) - necesidad_real

Un excedente/faltante menor a UN formato completo se considera redondeo normal
(tal como pide el brief) y no genera alerta.

Cada alerta trae, además del texto, un "score" de prioridad para poder mostrar
primero lo que la gerente realmente necesita revisar.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import forecasting
from .data_loader import BarrioPizzaData

SEVERITY_ORDER = {"CRITICO": 0, "ATENCION": 1, "OK": 2, "CALIDAD_DATOS": 0}


@dataclass
class Alerta:
    sucursal: str
    ingrediente_id: str
    nombre_ingrediente: str
    proveedor: str
    unidad_base: str
    formato_compra: str
    unidad_base_por_formato: float
    es_perecedero: bool

    proyeccion: float
    tendencia: str
    confianza: str
    error_backtest_pct: float | None
    semanas_excluidas: list

    stock_actual: float
    necesidad_real: float
    necesidad_formatos: int
    pedido_formatos: float
    pedido_base: float
    diff_base: float          # pedido - necesidad (en unidad base). + = de más, - = de menos
    cobertura_dias_con_pedido: float | None

    tipo: str        # "SOBRE_PEDIDO" | "SUB_PEDIDO" | "OLVIDO" | "OK"
    severidad: str    # "CRITICO" | "ATENCION" | "OK"
    prioridad_score: float
    mensaje: str
    sugerencia: str
    serie_historica: list = field(default_factory=list)


def _cobertura_dias(stock_mas_pedido: float, proyeccion_semanal: float) -> float | None:
    if proyeccion_semanal <= 0:
        return None
    consumo_diario = proyeccion_semanal / 7.0
    if consumo_diario <= 0:
        return None
    return stock_mas_pedido / consumo_diario


def calcular_alerta(data: BarrioPizzaData, sucursal: str, ingrediente_id: str) -> Alerta | None:
    meta = data.meta_ingrediente(ingrediente_id)
    if meta is None:
        return None  # se maneja aparte como problema de calidad de datos

    serie = data.serie_consumo(sucursal, ingrediente_id)
    if serie.empty:
        return None

    proy = forecasting.proyectar_consumo(
        serie["semana_num"].to_numpy(), serie["consumo_unidad_base"].to_numpy()
    )

    stock = data.stock_actual(sucursal, ingrediente_id)
    pedido_formatos = data.pedido_formatos(sucursal, ingrediente_id)
    formato_size = meta["unidad_base_por_formato"] or 1.0

    necesidad_real = max(proy.valor - stock, 0.0)
    necesidad_formatos = math.ceil(round(necesidad_real / formato_size, 6)) if formato_size > 0 else 0

    pedido_base = pedido_formatos * formato_size
    diff_base = pedido_base - necesidad_real

    cobertura = _cobertura_dias(stock + pedido_base, proy.valor)

    tolerancia = formato_size  # 1 formato de tolerancia = redondeo normal

    # ---- Clasificación ----
    if pedido_formatos == 0 and necesidad_formatos > 0:
        tipo = "OLVIDO"
    elif diff_base < -tolerancia:
        tipo = "SUB_PEDIDO"
    elif diff_base > tolerancia:
        tipo = "SOBRE_PEDIDO"
    else:
        tipo = "OK"

    pct_desvio = abs(diff_base) / max(necesidad_real, formato_size) * 100

    severidad = "OK"
    if tipo == "OLVIDO":
        severidad = "CRITICO"
    elif tipo == "SUB_PEDIDO":
        if (cobertura is not None and cobertura < 7) or pct_desvio > 40:
            severidad = "CRITICO"
        else:
            severidad = "ATENCION"
    elif tipo == "SOBRE_PEDIDO":
        if pct_desvio > 100 and meta["es_perecedero"]:
            severidad = "CRITICO"
        elif pct_desvio > 30:
            severidad = "ATENCION"
        else:
            severidad = "OK"
            tipo = "OK"

    # ---- Score de prioridad (mayor = revisar primero) ----
    score = pct_desvio
    if tipo == "OLVIDO":
        score += 60
    if meta["es_perecedero"] and tipo == "SOBRE_PEDIDO":
        score += 25
    if cobertura is not None and cobertura < 5:
        score += 40
    if proy.confianza == "Baja":
        score *= 0.85  # bajamos un poco la urgencia si la proyección es poco confiable

    # ---- Mensaje accionable ----
    nombre = meta["nombre"]
    unidad = meta["unidad_base"]
    if tipo == "OLVIDO":
        mensaje = (
            f"ALERTA: {sucursal} no pidió nada de {nombre} esta semana, pero se proyecta "
            f"un consumo de {proy.valor:.1f} {unidad} → riesgo de quiebre."
        )
        sugerencia = f"Agregar {necesidad_formatos} {meta['formato_compra']} de {nombre}."
    elif tipo == "SUB_PEDIDO":
        faltante = abs(diff_base)
        mensaje = (
            f"ALERTA: {sucursal} está pidiendo {faltante:.1f} {unidad} de {nombre} menos "
            f"que lo proyectado → riesgo de quiebre."
        )
        sugerencia = f"Subir el pedido a {necesidad_formatos} {meta['formato_compra']} (pidió {int(pedido_formatos)})."
    elif tipo == "SOBRE_PEDIDO":
        excedente = diff_base
        etiqueta_riesgo = "producto perecedero, riesgo de vencimiento" if meta["es_perecedero"] else "plata inmovilizada"
        mensaje = (
            f"ALERTA: {sucursal} está pidiendo {excedente:.1f} {unidad} de {nombre} de más "
            f"que lo proyectado → {etiqueta_riesgo}."
        )
        sugerencia = f"Bajar el pedido a {necesidad_formatos} {meta['formato_compra']} (pidió {int(pedido_formatos)})."
    else:
        mensaje = f"OK: {sucursal} - {nombre} está dentro de lo esperado (redondeo normal de formato)."
        sugerencia = "Sin acción necesaria."

    return Alerta(
        sucursal=sucursal,
        ingrediente_id=ingrediente_id,
        nombre_ingrediente=nombre,
        proveedor=meta["proveedor"],
        unidad_base=unidad,
        formato_compra=meta["formato_compra"],
        unidad_base_por_formato=formato_size,
        es_perecedero=meta["es_perecedero"],
        proyeccion=proy.valor,
        tendencia=proy.tendencia,
        confianza=proy.confianza,
        error_backtest_pct=proy.error_backtest_pct,
        semanas_excluidas=proy.semanas_excluidas,
        stock_actual=stock,
        necesidad_real=necesidad_real,
        necesidad_formatos=necesidad_formatos,
        pedido_formatos=pedido_formatos,
        pedido_base=pedido_base,
        diff_base=diff_base,
        cobertura_dias_con_pedido=cobertura,
        tipo=tipo,
        severidad=severidad,
        prioridad_score=round(score, 1),
        mensaje=mensaje,
        sugerencia=sugerencia,
        serie_historica=serie["consumo_unidad_base"].tolist(),
    )


def calcular_todas_las_alertas(data: BarrioPizzaData) -> pd.DataFrame:
    filas = []
    for sucursal, ingrediente_id in data.universo:
        alerta = calcular_alerta(data, sucursal, ingrediente_id)
        if alerta is not None:
            filas.append(alerta.__dict__)
    df = pd.DataFrame(filas)
    if df.empty:
        return df
    df["severidad_rank"] = df["severidad"].map(SEVERITY_ORDER)
    df = df.sort_values(["severidad_rank", "prioridad_score"], ascending=[True, False]).reset_index(drop=True)
    return df


def problemas_calidad_datos(data: BarrioPizzaData) -> pd.DataFrame:
    """Ingredientes pedidos que no existen en el maestro (ej. 'aji_chombo')."""
    return data.ingredientes_desconocidos.copy()
