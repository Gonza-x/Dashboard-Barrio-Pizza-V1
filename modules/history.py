"""
history.py
-----------
Historial de alertas y comparación entre semanas.

Cada vez que la gerente revisa una semana puede guardar una "foto" (snapshot)
de las alertas. Con eso el dashboard puede mostrar la evolución en el tiempo
(¿mejoramos respecto a la semana pasada?) y comparar dos semanas en detalle
(qué se resolvió, qué empeoró, qué sigue en alerta).

La persistencia es un simple archivo JSON en disco, para que el historial
sobreviva reinicios de la app. Se guardan solo las líneas EN ALERTA (no las
OK) más los conteos, lo justo para reconstruir la comparación.
"""
from __future__ import annotations

import json
import os
from datetime import datetime

import pandas as pd

HISTORIAL_PATH = os.path.join("data", "historial_alertas.json")

# Para comparar severidades: número menor = más grave.
_ORDEN = {"CRITICO": 0, "ATENCION": 1}


def resumen_semana(df_alertas: pd.DataFrame) -> dict:
    """Resume una semana: conteos por severidad + el detalle de las líneas
    que están en alerta (no OK), para poder comparar después."""
    if df_alertas is None or df_alertas.empty:
        return {"criticas": 0, "atencion": 0, "ok": 0, "alertas": []}

    conteo = df_alertas["severidad"].value_counts().to_dict()
    en_alerta = df_alertas[df_alertas["severidad"] != "OK"]
    alertas = [
        {
            "sucursal": r["sucursal"],
            "ingrediente": r["nombre_ingrediente"],
            "severidad": r["severidad"],
            "tipo": r["tipo"],
        }
        for _, r in en_alerta.iterrows()
    ]
    return {
        "criticas": int(conteo.get("CRITICO", 0)),
        "atencion": int(conteo.get("ATENCION", 0)),
        "ok": int(conteo.get("OK", 0)),
        "alertas": alertas,
    }


def cargar_historial() -> list[dict]:
    """Devuelve la lista de snapshots guardados (vieja → nueva)."""
    if not os.path.exists(HISTORIAL_PATH):
        return []
    try:
        with open(HISTORIAL_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _guardar_historial(historial: list[dict]) -> None:
    os.makedirs(os.path.dirname(HISTORIAL_PATH), exist_ok=True)
    with open(HISTORIAL_PATH, "w", encoding="utf-8") as f:
        json.dump(historial, f, ensure_ascii=False, indent=2)


def guardar_snapshot(df_alertas: pd.DataFrame, etiqueta: str | None = None) -> dict:
    """Agrega un snapshot de la semana actual al historial y lo devuelve."""
    historial = cargar_historial()
    ahora = datetime.now()
    snap = {
        "id": ahora.strftime("%Y%m%d%H%M%S"),
        "fecha": ahora.strftime("%d/%m/%Y %H:%M"),
        "etiqueta": (etiqueta or "").strip() or f"Semana del {ahora.strftime('%d/%m/%Y')}",
    }
    snap.update(resumen_semana(df_alertas))
    historial.append(snap)
    _guardar_historial(historial)
    return snap


def borrar_snapshot(snap_id: str) -> None:
    _guardar_historial([s for s in cargar_historial() if s.get("id") != snap_id])


def comparar(actual: dict, previo: dict) -> dict:
    """Compara la semana `actual` contra la `previo` (ambos resúmenes de
    resumen_semana / snapshots). Devuelve los deltas de conteo y el detalle de
    qué líneas se resolvieron, empeoraron o mejoraron."""
    def _key(linea):
        return (linea["sucursal"], linea["ingrediente"])

    cur = {_key(l): l for l in actual.get("alertas", [])}
    prev = {_key(l): l for l in previo.get("alertas", [])}

    resueltas, empeoraron, mejoraron = [], [], []

    # Resueltas: estaban en alerta antes y ahora ya no aparecen (pasaron a OK).
    for k, l in prev.items():
        if k not in cur:
            resueltas.append({
                "sucursal": l["sucursal"], "ingrediente": l["ingrediente"],
                "antes": l["severidad"], "ahora": "OK",
            })

    # Alertas actuales: nuevas, más graves o menos graves que antes.
    for k, l in cur.items():
        p = prev.get(k)
        antes = p["severidad"] if p else "OK"
        ahora = l["severidad"]
        if antes == ahora:
            continue
        item = {
            "sucursal": l["sucursal"], "ingrediente": l["ingrediente"],
            "antes": antes, "ahora": ahora,
        }
        # Nueva (antes OK) o subió de gravedad -> empeoró; si no, mejoró.
        if antes == "OK" or _ORDEN[ahora] < _ORDEN[antes]:
            empeoraron.append(item)
        else:
            mejoraron.append(item)

    return {
        "delta_criticas": actual.get("criticas", 0) - previo.get("criticas", 0),
        "delta_atencion": actual.get("atencion", 0) - previo.get("atencion", 0),
        "delta_ok": actual.get("ok", 0) - previo.get("ok", 0),
        "resueltas": resueltas,
        "empeoraron": empeoraron,
        "mejoraron": mejoraron,
    }