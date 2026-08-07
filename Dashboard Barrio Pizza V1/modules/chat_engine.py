"""
chat_engine.py
---------------
"Chat con los datos" usando tool use (function calling) de la API de Groq
(formato OpenAI). Soporta el mismo conjunto de herramientas.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from openai import OpenAI

# Modelo por defecto para Groq (compatible con tool calling)
MODEL = os.environ.get("BARRIO_PIZZA_CHAT_MODEL", "gpt-oss-20b")

SYSTEM_PROMPT = """Sos un asistente para la gerente de compras de Barrio Pizza,
una cadena de pizzerías en Panamá. Respondés preguntas sobre las órdenes de
compra de la semana (alertas, proyecciones, historial de consumo, anomalías
entre sucursales, pedido por proveedor), consultando SIEMPRE los datos reales
a través de las herramientas disponibles.

Cómo comportarte:
- Saludos y charla breve (hola, buenas, ¿cómo estás?, gracias, ¿qué podés
  hacer?): respondé de forma cálida y natural, en 1-2 líneas, y NO llames a
  ninguna herramienta. Presentate en una frase y ofrecé ayuda concreta. Por
  ejemplo: "¡Hola! Soy tu asistente de compras. Puedo revisar alertas de
  pedidos, anomalías entre sucursales o el pedido por proveedor. ¿Qué querés ver?"
- Preguntas sobre datos: usá las herramientas para obtener las cifras. Nunca
  inventes un número; si necesitás un dato, llamá a la herramienta que corresponda.
- Si una pregunta sobre datos no se puede responder con las herramientas
  disponibles, decilo con honestidad y sugerí qué sí podés consultar.

Estilo:
- Respondé siempre en español, en tono directo y accionable, como le hablarías
  a una gerente ocupada: la conclusión primero y el detalle después si hace falta.
- Sé breve: preferí 3-6 líneas o una lista corta antes que un párrafo largo.
"""


def _clean(obj):
    """Convierte tipos numpy/pandas a tipos nativos serializables en JSON."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return round(float(obj), 3)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, float):
        return round(obj, 3)
    return obj


# Herramientas en formato OpenAI
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "buscar_alertas",
            "description": (
                "Busca alertas de pedidos de compra. Podés filtrar por sucursal, "
                "ingrediente (nombre o id), severidad (CRITICO/ATENCION/OK) o tipo "
                "(OLVIDO/SUB_PEDIDO/SOBRE_PEDIDO/OK). Dejá un filtro vacío para no aplicarlo."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sucursal": {"type": "string", "description": "Nombre exacto o parcial de la sucursal"},
                    "ingrediente": {"type": "string", "description": "Nombre o id del ingrediente"},
                    "severidad": {"type": "string", "enum": ["CRITICO", "ATENCION", "OK"]},
                    "tipo": {"type": "string", "enum": ["OLVIDO", "SUB_PEDIDO", "SOBRE_PEDIDO", "OK"]},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "resumen_general",
            "description": "Devuelve el resumen ejecutivo: cantidad de alertas críticas, de atención, sucursales con problemas críticos, etc.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "historial_consumo",
            "description": "Devuelve el consumo semanal histórico (6 semanas) de un ingrediente en una sucursal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sucursal": {"type": "string"},
                    "ingrediente": {"type": "string"},
                },
                "required": ["sucursal", "ingrediente"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "anomalias_entre_sucursales",
            "description": "Devuelve anomalías entre sucursales de DOS tipos: (1) por consumo/caja, cuando una sucursal usa un ingrediente por pizza de forma muy distinta al resto; y (2) por cobertura en semanas, cuando una sucursal pide muchas más o muchas menos semanas de stock que las demás. Devuelve ambas listas; si una viene vacía, es que no hubo anomalías de ese tipo.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pedido_por_proveedor",
            "description": "Devuelve el pedido corregido (recomendado) agrupado por proveedor. Podés filtrar por nombre de proveedor.",
            "parameters": {
                "type": "object",
                "properties": {"proveedor": {"type": "string"}},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "problemas_calidad_datos",
            "description": "Devuelve ingredientes pedidos que no existen en el maestro de ingredientes (ej. errores de tipeo, insumos nuevos sin dar de alta).",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
]


class ChatEngine:
    def __init__(self, data, df_alertas: pd.DataFrame, anomaly_df: pd.DataFrame, pedido_df: pd.DataFrame):
        self.data = data
        self.df_alertas = df_alertas
        self.anomaly_df = anomaly_df
        self.pedido_df = pedido_df

    # ------------------------------------------------------------------
    # Implementación real de cada tool, sobre los DataFrames ya calculados
    # ------------------------------------------------------------------
    def _tool_buscar_alertas(self, sucursal=None, ingrediente=None, severidad=None, tipo=None):
        df = self.df_alertas.copy()
        if sucursal:
            df = df[df["sucursal"].str.contains(sucursal, case=False, na=False)]
        if ingrediente:
            mask = (
                df["nombre_ingrediente"].str.contains(ingrediente, case=False, na=False)
                | df["ingrediente_id"].str.contains(ingrediente, case=False, na=False)
            )
            df = df[mask]
        if severidad:
            df = df[df["severidad"] == severidad]
        if tipo:
            df = df[df["tipo"] == tipo]
        cols = ["sucursal", "nombre_ingrediente", "tipo", "severidad", "proyeccion",
                "stock_actual", "pedido_formatos", "necesidad_formatos", "mensaje", "sugerencia"]
        df = df[cols].head(40)
        return _clean(df.to_dict(orient="records"))

    def _tool_resumen_general(self):
        from . import insights
        return _clean(insights.resumen_ejecutivo(self.df_alertas))

    def _tool_historial_consumo(self, sucursal: str, ingrediente: str):
        ids = self.data.ingredientes[
            self.data.ingredientes["nombre"].str.contains(ingrediente, case=False, na=False)
            | self.data.ingredientes["ingrediente_id"].str.contains(ingrediente, case=False, na=False)
        ]["ingrediente_id"].tolist()
        if not ids:
            return {"error": f"No se encontró el ingrediente '{ingrediente}'"}
        ingrediente_id = ids[0]
        serie = self.data.serie_consumo(sucursal, ingrediente_id)
        return _clean({
            "sucursal": sucursal,
            "ingrediente": ingrediente_id,
            "consumo_por_semana": serie.set_index("semana_num")["consumo_unidad_base"].to_dict(),
        })

    def _tool_anomalias_entre_sucursales(self):
        """Reúne los DOS tipos de anomalía entre sucursales para que el chat
        cubra lo mismo que la pestaña visual:
        - por consumo/caja (receta): usa un ingrediente por pizza distinto al resto.
        - por cobertura en semanas (stock): pide muchas más/menos semanas que el resto.
        """
        from . import anomaly
        ratio = [] if self.anomaly_df.empty else _clean(self.anomaly_df.to_dict(orient="records"))
        try:
            cob_df = anomaly.detectar_anomalias_por_cobertura(
                self.data, self.df_alertas, umbral_multiplo=1.3
            )
            cobertura = [] if cob_df.empty else _clean(cob_df.to_dict(orient="records"))
        except Exception as exc:
            cobertura = {"error": str(exc)}
        return {
            "anomalias_por_consumo_caja": ratio,
            "anomalias_por_cobertura_semanas": cobertura,
        }

    def _tool_pedido_por_proveedor(self, proveedor=None):
        df = self.pedido_df.copy()
        if proveedor:
            df = df[df["proveedor"].str.contains(proveedor, case=False, na=False)]
        return _clean(df.head(60).to_dict(orient="records"))

    def _tool_problemas_calidad_datos(self):
        from . import alerts
        return _clean(alerts.problemas_calidad_datos(self.data).to_dict(orient="records"))

    def _dispatch(self, name: str, tool_input: dict):
        fn_map = {
            "buscar_alertas": self._tool_buscar_alertas,
            "resumen_general": self._tool_resumen_general,
            "historial_consumo": self._tool_historial_consumo,
            "anomalias_entre_sucursales": self._tool_anomalias_entre_sucursales,
            "pedido_por_proveedor": self._tool_pedido_por_proveedor,
            "problemas_calidad_datos": self._tool_problemas_calidad_datos,
        }
        fn = fn_map.get(name)
        if fn is None:
            return {"error": f"Herramienta desconocida: {name}"}
        try:
            return fn(**tool_input)
        except Exception as exc:
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    def responder(self, pregunta: str, historial: list[dict] | None = None) -> tuple[str, list[dict]]:
        """
        Devuelve (respuesta_texto, historial_actualizado).
        `historial` es una lista de mensajes en formato OpenAI (role: user/assistant/tool).
        """
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            return (
                "No hay una GROQ_API_KEY configurada. Definila como variable de entorno (en el archivo .env) para activar el chat con IA.",
                historial or [],
            )

        client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )

        messages = list(historial or [])
        messages.append({"role": "user", "content": pregunta})

        for _ in range(6):
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    max_tokens=1024,
                    messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
                    tools=TOOLS,
                    tool_choice="auto",
                )
            except Exception as e:
                # Capturamos cualquier error de la API (falta de fondos, clave inválida, etc.)
                error_msg = str(e)
                if "insufficent_credits" in error_msg or "OutOfFundsException" in error_msg:
                    return ("No tengo créditos disponibles para responder. Si quieres usar la IA, recarga saldo en Groq. Mientras tanto, puedo responder con el modo simplificado (basado en reglas).", historial)
                elif "Authentication" in error_msg:
                    return ("La API Key de Groq es inválida o no está configurada correctamente. Revisa tu archivo .env.", historial)
                else:
                    return (f"Error inesperado en la IA: {error_msg}. Volviendo al modo simplificado.", historial)

            response_message = response.choices[0].message

            # Si no hay llamadas a herramientas, devolvemos la respuesta textual
            if not response_message.tool_calls:
                texto = response_message.content or "¡Hola! ¿En qué te ayudo con las compras de esta semana?"
                messages.append({"role": "assistant", "content": texto})
                return texto, messages

            # Agregamos la respuesta del asistente (con las tool_calls) como dict
            # plano: serializable en JSON, accesible con ["..."] y compatible con la API.
            messages.append({
                "role": "assistant",
                "content": response_message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in response_message.tool_calls
                ],
            })

            tool_results = []
            for tool_call in response_message.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                result = self._dispatch(name, args)
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, ensure_ascii=False),
                })

            # Agregamos los resultados de las herramientas a la conversación
            messages.extend(tool_results)

        return "No pude completar la respuesta (demasiadas consultas encadenadas).", messages