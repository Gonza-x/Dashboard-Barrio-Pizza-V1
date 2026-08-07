"""
data_loader.py
----------------
Carga los 4 CSV de Barrio Pizza y los normaliza en una única tabla "maestra"
por (sucursal, ingrediente) con:
  - la serie histórica de consumo (6 semanas, en unidad base)
  - el inventario actual (en unidad base)
  - el pedido de la semana (convertido de formatos a unidad base)
  - la metadata del ingrediente (proveedor, formato de compra, perecedero, etc.)

También detecta problemas de calidad de datos:
  - ingredientes pedidos que no existen en el maestro de ingredientes
    (ej. "aji_chombo" en orden_compra_semana.csv)
  - ingredientes con inventario/consumo pero SIN pedido esta semana
    (posible olvido -> se marca cantidad_formatos = 0, no se descarta)
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd


REQUIRED_INGREDIENTE_COLS = [
    "ingrediente_id", "nombre", "proveedor", "unidad_base",
    "formato_compra", "unidad_base_por_formato", "es_perecedero",
]

# Diccionario de correcciones de nombres de ingredientes
CORRECCIONES_NOMBRES = {
    "pina": "Piña",
    "harina00": "Harina",
    # Agrega aquí más correcciones si aparecen
}


def _norm_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(["si", "sí", "true", "1", "yes"])


def _limpiar_nombre(nombre: str) -> str:
    """
    Limpia y normaliza el nombre de un ingrediente.
    1. Elimina números al final (ej. 'harina00' -> 'harina').
    2. Aplica correcciones del diccionario (ej. 'pina' -> 'Piña').
    3. Capitaliza correctamente (primera letra mayúscula de cada palabra).
    """
    if pd.isna(nombre):
        return ""
    
    nombre = str(nombre).strip()
    
    # 1. Eliminar dígitos al final de la cadena (ej. 'harina00' -> 'harina')
    nombre = re.sub(r'\d+$', '', nombre).strip()
    
    # 2. Aplicar correcciones específicas del diccionario
    nombre_lower = nombre.lower()
    if nombre_lower in CORRECCIONES_NOMBRES:
        nombre = CORRECCIONES_NOMBRES[nombre_lower]
    
    # 3. Capitalizar (primera letra mayúscula de cada palabra)
    return ' '.join(palabra.capitalize() for palabra in nombre.split())


def load_ingredientes(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    missing = set(REQUIRED_INGREDIENTE_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"ingredientes.csv no tiene las columnas esperadas: {missing}")
    
    df["ingrediente_id"] = df["ingrediente_id"].str.strip()
    df["nombre"] = df["nombre"].apply(_limpiar_nombre)  # <--- APLICAMOS LA LIMPIEZA AQUÍ
    
    df["es_perecedero"] = _norm_bool(df["es_perecedero"])
    df["unidad_base_por_formato"] = pd.to_numeric(df["unidad_base_por_formato"], errors="coerce")
    return df


def load_inventario(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    df["sucursal"] = df["sucursal"].str.strip()
    df["ingrediente_id"] = df["ingrediente_id"].str.strip()
    df["stock_actual_unidad_base"] = pd.to_numeric(df["stock_actual_unidad_base"], errors="coerce")
    return df


def load_consumo(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    df["sucursal"] = df["sucursal"].str.strip()
    df["ingrediente_id"] = df["ingrediente_id"].str.strip()
    df["semana"] = df["semana"].str.strip()
    df["semana_num"] = df["semana"].str.extract(r"(\d+)").astype(int)
    df["consumo_unidad_base"] = pd.to_numeric(df["consumo_unidad_base"], errors="coerce")
    return df


def load_orden(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    df["sucursal"] = df["sucursal"].str.strip()
    df["ingrediente_id"] = df["ingrediente_id"].str.strip()
    df["cantidad_formatos"] = pd.to_numeric(df["cantidad_formatos"], errors="coerce")
    return df


class BarrioPizzaData:
    """Contenedor central de datos, ya limpios y cruzados."""

    def __init__(self, ingredientes: pd.DataFrame, inventario: pd.DataFrame,
                 consumo: pd.DataFrame, orden: pd.DataFrame):
        self.ingredientes = ingredientes
        self.inventario = inventario
        self.consumo = consumo
        self.orden_raw = orden

        self.sucursales = sorted(
            set(inventario["sucursal"]) | set(consumo["sucursal"]) | set(orden["sucursal"])
        )

        # --- Calidad de datos: ingredientes pedidos que no existen en el maestro ---
        ids_validos = set(ingredientes["ingrediente_id"])
        self.orden_raw["ingrediente_reconocido"] = self.orden_raw["ingrediente_id"].isin(ids_validos)
        self.ingredientes_desconocidos = (
            self.orden_raw.loc[~self.orden_raw["ingrediente_reconocido"]]
            [["sucursal", "ingrediente_id", "cantidad_formatos"]]
            .reset_index(drop=True)
        )

        # Orden válida (ingredientes reconocidos), con conversión a unidad base
        orden_valida = self.orden_raw.loc[self.orden_raw["ingrediente_reconocido"]].merge(
            ingredientes[["ingrediente_id", "unidad_base_por_formato"]],
            on="ingrediente_id", how="left",
        )
        orden_valida["cantidad_unidad_base"] = (
            orden_valida["cantidad_formatos"] * orden_valida["unidad_base_por_formato"]
        )
        self.orden = orden_valida

        # --- Universo completo de combinaciones (sucursal, ingrediente) esperadas ---
        combos_inventario = set(map(tuple, inventario[["sucursal", "ingrediente_id"]].values))
        combos_consumo = set(map(tuple, consumo[["sucursal", "ingrediente_id"]].values))
        self.universo = sorted(combos_inventario | combos_consumo)

    # ------------------------------------------------------------------
    def serie_consumo(self, sucursal: str, ingrediente_id: str) -> pd.DataFrame:
        sub = self.consumo[
            (self.consumo["sucursal"] == sucursal) & (self.consumo["ingrediente_id"] == ingrediente_id)
        ].sort_values("semana_num")
        return sub[["semana_num", "consumo_unidad_base"]].reset_index(drop=True)

    def stock_actual(self, sucursal: str, ingrediente_id: str) -> float:
        sub = self.inventario[
            (self.inventario["sucursal"] == sucursal) & (self.inventario["ingrediente_id"] == ingrediente_id)
        ]
        if sub.empty:
            return 0.0
        return float(sub["stock_actual_unidad_base"].iloc[0])

    def pedido_formatos(self, sucursal: str, ingrediente_id: str) -> float:
        sub = self.orden[
            (self.orden["sucursal"] == sucursal) & (self.orden["ingrediente_id"] == ingrediente_id)
        ]
        if sub.empty:
            return 0.0  # no pidieron nada de este insumo -> posible olvido
        return float(sub["cantidad_formatos"].iloc[0])

    def meta_ingrediente(self, ingrediente_id: str) -> dict | None:
        sub = self.ingredientes[self.ingredientes["ingrediente_id"] == ingrediente_id]
        if sub.empty:
            return None
        row = sub.iloc[0]
        return {
            "nombre": row["nombre"],
            "proveedor": row["proveedor"],
            "unidad_base": row["unidad_base"],
            "formato_compra": row["formato_compra"],
            "unidad_base_por_formato": float(row["unidad_base_por_formato"]),
            "es_perecedero": bool(row["es_perecedero"]),
        }

    def set_orden(self, nueva_orden: pd.DataFrame):
        """Permite reemplazar la orden (para el simulador / edición en vivo)."""
        nueva_orden = nueva_orden.copy()
        ids_validos = set(self.ingredientes["ingrediente_id"])
        nueva_orden["ingrediente_reconocido"] = nueva_orden["ingrediente_id"].isin(ids_validos)
        self.ingredientes_desconocidos = (
            nueva_orden.loc[~nueva_orden["ingrediente_reconocido"]]
            [["sucursal", "ingrediente_id", "cantidad_formatos"]]
            .reset_index(drop=True)
        )
        orden_valida = nueva_orden.loc[nueva_orden["ingrediente_reconocido"]].merge(
            self.ingredientes[["ingrediente_id", "unidad_base_por_formato"]],
            on="ingrediente_id", how="left",
        )
        orden_valida["cantidad_unidad_base"] = (
            orden_valida["cantidad_formatos"] * orden_valida["unidad_base_por_formato"]
        )
        self.orden_raw = nueva_orden
        self.orden = orden_valida


def load_all(data_dir: str) -> BarrioPizzaData:
    ingredientes = load_ingredientes(f"{data_dir}/ingredientes.csv")
    inventario = load_inventario(f"{data_dir}/inventario_actual.csv")
    consumo = load_consumo(f"{data_dir}/consumo_historico.csv")
    orden = load_orden(f"{data_dir}/orden_compra_semana.csv")
    return BarrioPizzaData(ingredientes, inventario, consumo, orden)