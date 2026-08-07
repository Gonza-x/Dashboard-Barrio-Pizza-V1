"""
insights.py
------------
Agregaciones de alto nivel para la vista ejecutiva y el pedido corregido
agrupado por proveedor (para reenviar directo a cada uno).
"""
from __future__ import annotations

import pandas as pd

from .data_loader import BarrioPizzaData


def resumen_ejecutivo(df_alertas: pd.DataFrame) -> dict:
    if df_alertas.empty:
        return {
            "total_criticas": 0, "total_atencion": 0, "total_ok": 0,
            "sucursales_con_criticas": [], "olvidos": 0,
            "sobre_pedidos_perecederos": 0,
        }
    criticas = df_alertas[df_alertas["severidad"] == "CRITICO"]
    atencion = df_alertas[df_alertas["severidad"] == "ATENCION"]
    return {
        "total_criticas": len(criticas),
        "total_atencion": len(atencion),
        "total_ok": len(df_alertas[df_alertas["severidad"] == "OK"]),
        "sucursales_con_criticas": sorted(criticas["sucursal"].unique().tolist()),
        "olvidos": int((df_alertas["tipo"] == "OLVIDO").sum()),
        "sub_pedidos": int((df_alertas["tipo"] == "SUB_PEDIDO").sum()),
        "sobre_pedidos": int((df_alertas["tipo"] == "SOBRE_PEDIDO").sum()),
        "sobre_pedidos_perecederos": int(
            ((df_alertas["tipo"] == "SOBRE_PEDIDO") & (df_alertas["es_perecedero"])).sum()
        ),
    }


def pedido_corregido(data: BarrioPizzaData, df_alertas: pd.DataFrame) -> pd.DataFrame:
    """
    Pedido recomendado = necesidad_formatos (ya redondeada hacia arriba al
    formato completo). Para lo que ya estaba OK, coincide (o casi) con lo
    que la sucursal ya había pedido.
    """
    if df_alertas.empty:
        return pd.DataFrame()
    cols = ["sucursal", "proveedor", "nombre_ingrediente", "ingrediente_id",
            "formato_compra", "pedido_formatos", "necesidad_formatos", "tipo", "severidad"]
    out = df_alertas[cols].copy()
    out = out.rename(columns={
        "pedido_formatos": "pedido_original_formatos",
        "necesidad_formatos": "pedido_recomendado_formatos",
    })
    out["cambio"] = out["pedido_recomendado_formatos"] - out["pedido_original_formatos"]
    return out.sort_values(["proveedor", "sucursal", "nombre_ingrediente"]).reset_index(drop=True)


def texto_por_proveedor(pedido_df: pd.DataFrame) -> dict[str, str]:
    """Genera un texto tipo mensaje, uno por proveedor, listo para reenviar."""
    textos = {}
    if pedido_df.empty:
        return textos
    for proveedor, grupo in pedido_df.groupby("proveedor"):
        lineas = [f"Pedido corregido - {proveedor}", "-" * 40]
        for sucursal, sub in grupo.groupby("sucursal"):
            lineas.append(f"\n{sucursal}:")
            for _, row in sub.iterrows():
                cant = int(row["pedido_recomendado_formatos"])
                if cant <= 0:
                    continue
                marca = ""
                if row["cambio"] > 0:
                    marca = f"  (+{int(row['cambio'])} vs. pedido original)"
                elif row["cambio"] < 0:
                    marca = f"  ({int(row['cambio'])} vs. pedido original)"
                lineas.append(f"  - {int(cant)} x {row['formato_compra']} de {row['nombre_ingrediente']}{marca}")
        textos[proveedor] = "\n".join(lineas)
    return textos
