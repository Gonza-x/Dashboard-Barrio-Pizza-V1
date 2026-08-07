"""
anomaly.py
-----------
Detección de pedidos "raros" comparando una sucursal contra las demás.

Comparar consumo crudo entre sucursales es injusto: una sucursal más grande
consume más de todo. Usamos "cajas_pizza" (proxy de volumen de venta / cantidad
de pizzas despachadas) para normalizar, y comparamos la RATIO
  (consumo del ingrediente / cajas_pizza)
entre sucursales. Así detectamos, por ejemplo, una sucursal que usa mucho más
queso por pizza que el resto — que es la señal de "algo raro" real, no solo
de que esa sucursal vende más.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data_loader import BarrioPizzaData

PROXY_INGREDIENTE = "cajas_pizza"


def tabla_ratios(data: BarrioPizzaData) -> pd.DataFrame:
    consumo = data.consumo
    # Usamos MEDIANA en vez de promedio: una sola semana atípica (ej. un catering
    # puntual) no debe disparar una falsa anomalía de "esta sucursal siempre pide
    # distinto" -- para eso ya está la detección de semanas atípicas en forecasting.
    proxy = (
        consumo[consumo["ingrediente_id"] == PROXY_INGREDIENTE]
        .groupby("sucursal")["consumo_unidad_base"].median()
        .rename("proxy_promedio")
    )

    prom_ingrediente = (
        consumo[consumo["ingrediente_id"] != PROXY_INGREDIENTE]
        .groupby(["sucursal", "ingrediente_id"])["consumo_unidad_base"].median()
        .reset_index()
        .rename(columns={"consumo_unidad_base": "consumo_promedio"})
    )

    tabla = prom_ingrediente.merge(proxy, on="sucursal", how="left")
    tabla["ratio_por_caja"] = tabla["consumo_promedio"] / tabla["proxy_promedio"]
    return tabla


def detectar_anomalias(data: BarrioPizzaData, umbral_desvio_pct: float = 15.0) -> pd.DataFrame:
    """
    Para cada ingrediente, compara la ratio de cada sucursal contra la MEDIANA
    de las otras sucursales. Si se desvía más que `umbral_desvio_pct`, se marca.
    Con solo 4 sucursales evitamos z-scores clásicos (poca muestra) y usamos
    desviación porcentual contra la mediana de los pares, que es más robusto.
    """
    tabla = tabla_ratios(data)
    filas = []
    for ingrediente_id, grupo in tabla.groupby("ingrediente_id"):
        if len(grupo) < 2:
            continue
        for _, row in grupo.iterrows():
            otras = grupo.loc[grupo["sucursal"] != row["sucursal"], "ratio_por_caja"]
            if otras.empty or otras.median() == 0:
                continue
            mediana_otras = otras.median()
            desvio_pct = (row["ratio_por_caja"] - mediana_otras) / mediana_otras * 100
            if abs(desvio_pct) >= umbral_desvio_pct:
                filas.append({
                    "sucursal": row["sucursal"],
                    "ingrediente_id": ingrediente_id,
                    "ratio_por_caja": row["ratio_por_caja"],
                    "mediana_otras_sucursales": mediana_otras,
                    "desvio_pct": desvio_pct,
                    "direccion": "por encima" if desvio_pct > 0 else "por debajo",
                })
    df = pd.DataFrame(filas)
    if df.empty:
        return df
    ingredientes_meta = data.ingredientes.set_index("ingrediente_id")[["nombre"]]
    df = df.merge(ingredientes_meta, on="ingrediente_id", how="left")
    df["desvio_abs"] = df["desvio_pct"].abs()
    df = df.sort_values("desvio_abs", ascending=False).drop(columns="desvio_abs").reset_index(drop=True)
    return df


# ================================================================
# NUEVA FUNCIÓN: Detección de anomalías por cobertura en semanas
# ================================================================

def detectar_anomalias_por_cobertura(data: BarrioPizzaData, df_alertas: pd.DataFrame, umbral_multiplo: float = 1.3) -> pd.DataFrame:
    """
    Detecta si una sucursal pidió un insumo en una cantidad muy distinta a la que pidieron
    las demás sucursales, medido en "semanas de cobertura" (pedido / consumo semanal).
    
    Ejemplo: Si la mediana de cobertura es 2 semanas y una sucursal pidió para 8 semanas,
    el múltiplo es 4x → anomalía.
    """
    if df_alertas.empty:
        return pd.DataFrame()

    # 1. Tomamos los datos necesarios del DataFrame de alertas
    df = df_alertas[['sucursal', 'ingrediente_id', 'nombre_ingrediente', 'proyeccion', 
                     'pedido_base', 'formato_compra', 'pedido_formatos', 'tipo']].copy()
    
    # Evitamos división por cero si la proyección es 0
    df['consumo_semanal'] = df['proyeccion'].apply(lambda x: max(x, 0.1))
    
    # 2. Calcular las semanas de cobertura que tiene cada sucursal con su pedido actual
    df['semanas_cobertura'] = df['pedido_base'] / df['consumo_semanal']
    
    # 3. Para cada ingrediente, calcular la MEDIANA de cobertura de las OTRAS sucursales
    filas = []
    for ingrediente_id, grupo in df.groupby('ingrediente_id'):
        if len(grupo) < 2:
            continue
            
        for _, row in grupo.iterrows():
            sucursal_actual = row['sucursal']
            cobertura_actual = row['semanas_cobertura']
            
            # Filtrar solo las otras sucursales
            otras = grupo[grupo['sucursal'] != sucursal_actual]
            if otras.empty:
                continue
                
            mediana_otras = otras['semanas_cobertura'].median()
            if mediana_otras == 0:
                continue
                
            # 4. Calcular el múltiplo
            multiplo = cobertura_actual / mediana_otras
            
            # Anomalía si es mayor al umbral (ej. 1.3x) o menor a 1/umbral (ej. 0.7x)
            if multiplo >= umbral_multiplo or multiplo <= (1 / umbral_multiplo):
                # Calcular cuánto debería haber pedido para igualar la mediana
                pedido_esperado_base = mediana_otras * row['consumo_semanal']
                diferencia_base = row['pedido_base'] - pedido_esperado_base
                
                # Convertir diferencia a formatos (para el mensaje)
                meta = data.meta_ingrediente(ingrediente_id)
                formato_size = meta['unidad_base_por_formato'] if meta else 1.0
                diferencia_formatos = round(diferencia_base / formato_size)
                
                filas.append({
                    'sucursal': sucursal_actual,
                    'ingrediente_id': ingrediente_id,
                    'nombre_ingrediente': row['nombre_ingrediente'],
                    'formato_compra': row['formato_compra'],
                    'semanas_cobertura_actual': round(cobertura_actual, 1),
                    'semanas_cobertura_mediana': round(mediana_otras, 1),
                    'multiplo': round(multiplo, 1),
                    'pedido_actual_formatos': int(row['pedido_formatos']),
                    'pedido_recomendado_formatos': round(pedido_esperado_base / formato_size),
                    'diferencia_formatos': int(diferencia_formatos),
                    'direccion': 'EXCESO' if multiplo > 1 else 'FALTANTE',
                    'tipo_alerta_original': row['tipo']  # Para saber si ya tenía una alerta
                })
    
    df_resultado = pd.DataFrame(filas)
    if df_resultado.empty:
        return df_resultado
    
    # Ordenar por mayor múltiplo (más grave)
    df_resultado = df_resultado.sort_values('multiplo', ascending=False).reset_index(drop=True)
    return df_resultado