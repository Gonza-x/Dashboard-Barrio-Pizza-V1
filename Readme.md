<div align="center">
  <img src="assets/logo%20barrio%20pizza.jpg" alt="Barrio Pizza" width="130">

# Barrio Pizza — Revisor Inteligente de Órdenes de Compra

  <em>Revisa automáticamente las órdenes de compra semanales y detecta pedidos de más, de menos y olvidos.</em>
</div>

Dashboard que **revisa automáticamente las órdenes de compra semanales** de las sucursales de Barrio Pizza detectando si están pidiendo de más, de menos, o si se olvidaron de algún insumo, y propone el pedido corregido. Todo de un vistazo sin que la gerente de compras tenga que leer tablas crudas ni revisar producto por producto.

Está construido con **Streamlit** e incluye proyección de demanda, un motor de alertas, detección de anomalías entre sucursales, un chat con IA conectado a los datos reales, el pedido corregido agrupado por proveedor, carga de órdenes desde archivo y un historial que compara semana contra semana.

**Demo en vivo:** [Abrir el dashboard](https://dashboard-barrio-pizza-v1-dstdmmfstzbkkhguy6nj9n.streamlit.app/)

---

## Problematica

Barrio Pizza tiene 10 sucursales en Panamá (este entregable trabaja con los datos de 4), cada semana, cada sucursal arma su orden de compra. Cuando piden **de más** inmovilizan dinero y se les vence producto y cuando piden **de menos** se quedan sin insumo en pleno servicio. Hoy esas órdenes se aprueban "al ojo" lo que consume mucho tiempo de la gerente de compras y es propenso a errores.

La visión es que la gerente cargue las órdenes de la semana y la herramienta le devuelva las alertas al instante por lo que este dashboard es esa herramienta.

---

## Instalación y ejecución

**Requisitos:** Python 3.10 o superior.

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Configurar el chat con IA (opcional, ver nota abajo)
cp .env.example .env
#   Editar .env y pegar tu GROQ_API_KEY
#   Es gratis y sin tarjeta: https://console.groq.com/keys

# 3. Ejecutar
python -m streamlit run app.py
```

La app abre en `http://localhost:8501`.

> **Nota sobre la API key:** todo el dashboard funciona **sin** ninguna clave (KPIs, alertas, gráficos, anomalías, simulador e historial). Lo único que necesita una clave es el *Chat con los datos* que usa un modelo de IA. Sin clave el chat cae automáticamente a un modo simplificado basado en reglas. La clave de Groq es gratuita.

---

## Recorrido por la app

- **Panel:** Resumen ejecutivo con KPIs (alertas críticas, de atención, correctas, mejor y peor sucursal), la tabla de alertas filtrable, y gráficos de dónde se concentran los problemas.
- **Proveedores:** El pedido corregido agrupado por proveedor, con un texto listo para copiar y reenviarle a cada uno su parte.
- **Anomalías:** Comparación entre sucursales para detectar pedidos raros, con dos enfoques (por consumo/caja y por cobertura en semanas).
- **Simulador:** Editar las cantidades a mano **o subir un CSV de órdenes** y ver las alertas recalcularse al instante, acercándose a la visión final de la herramienta.
- **Historial:** Guardar una "foto" de las alertas de cada semana y comparar la semana actual contra una anterior: qué se resolvió, qué empeoró y qué sigue en alerta.
- **Chat:** Preguntas en español por ejemplo:"¿qué sucursal está pidiendo demasiado queso esta semana?" respondidas en texto con IA conectada a los datos reales.

---

## Cómo funciona (decisiones técnicas)

### 1. Unidades:

El consumo y el inventario vienen en **unidad base** (kg, L, unidades) pero las órdenes vienen en **formatos** (sacos, cajas, bidones). Todo el pipeline convierte usando `unidad_base_por_formato` del maestro de ingredientes.

Como los insumos solo se compran en formatos completos (no existe medio saco), la necesidad se redondea **hacia arriba** con `ceil`. Y siguiendo el brief, un excedente o faltante **menor a un formato completo** se considera redondeo normal y **no** genera alerta, por lo que la tolerancia es exactamente un formato, evitando asi una lluvia de falsos positivos.

### 2. Proyección de demanda (`forecasting.py`)

Con solo 6 semanas de histórico un modelo complejo (ARIMA, Prophet) sobreajusta y se vuelve una caja negra imposible de explicar. Opté por un método **robusto y explicable** que hace las dos cosas que el brief valora: captar tendencia *e* ignorar semanas atípicas:

- **Detección de semanas atípicas con MAD:** A diferencia del z-score clásico con media y desvío estándar donde el propio outlier distorsiona, la mediana y la desviación absoluta mediana resisten un catering puntual o una semana rara.
- **Tendencia con Theil-Sen:** Capta crecimiento o caída sostenida sin dejarse arrastrar por un solo dato, mucho más robusto que una regresión lineal común.
- **Backtesting honesto:** re-proyecto la última semana conocida usando solo las anteriores y mido el error con ello puedo definir la **confianza** (Alta / Media / Baja) que se muestra en el dashboard, en vez de un número falsamente preciso y un pedido con proyección poco confiable baja su prioridad automáticamente.

Hay fallbacks para los casos límite: con menos de 3 semanas no se detectan outliers de forma fiable y si excluir atípicas deja muy pocos puntos no se excluyen.

### 3. Motor de alertas (`alerts.py`)

Para cada par (sucursal, ingrediente):

```
necesidad_real = max(consumo_proyectado − inventario_actual, 0)
necesidad_formatos = ceil(necesidad_real / tamaño_formato)
diferencia = pedido − necesidad_real   (en unidad base)
```

Con eso se clasifica cada línea en **OLVIDO** (no pidió nada pero necesita), **SUB_PEDIDO** (faltante mayor a un formato), **SOBRE_PEDIDO** (excedente mayor a un formato) u **OK**.

La **severidad:** Un olvido es siempre crítico; un sub-pedido es crítico si la cobertura cae por debajo de una semana o el desvío supera el 40%; un sobre-pedido es crítico solo si el producto es perecedero y el exceso es grande (riesgo de vencimiento), y leve si es dinero inmovilizado.

Además cada alerta lleva un **score de prioridad** para que la gerente vea primero lo que de verdad importa: suma peso por olvidos, por sobre-stock de perecederos y por cobertura baja, y resta urgencia cuando la proyección es poco confiable. Los mensajes siguen el formato accionable del brief:

> *ALERTA: Costa del Este está pidiendo 12.0 kg de mozzarella menos que lo proyectado → riesgo de quiebre.*

### 4. Detección de anomalías entre sucursales (`anomaly.py)

Comparar consumo crudo entre sucursales es injusto proque una sucursal más grande consume más de todo. Por eso opte por normalización por `cajas_pizza` (proxy del volumen de venta) y comparo la **ratio** `consumo / cajas` —lo que me permite detectar si una sucursal usa más queso *por pizza* que el resto, que es la señal real de "algo raro", no solo de que vende más.

Con solo 4 sucursales evité z-scores clásicos (poca muestra) y comparo cada sucursal contra la **mediana de las otras** que es más robusta. Hay dos visiones complementarias:

- **Por consumo/caja (receta):** ¿una sucursal usa un insumo por pizza muy distinto al resto? Señal de receta mal aplicada o desperdicio.
- **Por cobertura en semanas (stock):** ¿una sucursal pidió muchas más o muchas menos semanas de stock que las demás? Señal de sobre-compra u olvido.

### 5. Datos incompletos y calidad de datos

Las órdenes reales traen ingredientes que no existen en el maestro (errores de tipeo, insumos nuevos sin dar de alta). En vez de romper silenciosamente, el cargador los separa en `ingredientes_desconocidos` y el dashboard los muestra aparte con una sugerencia de acción. Es un caso real de datos sucios, tratado explícitamente.

### 6. Chat con los datos (`chat_engine.py`)

Un cuadro donde la gerente escribe en español y recibe la respuesta en texto. Por dentro usa **Groq** (endpoint compatible con OpenAI) con **tool calling** donde el modelo como tal no ve las tablas crudas, sino que llama a herramientas que corren sobre los DataFrames ya calculados y le devuelven **cifras reales**. Así nunca inventa números  y si necesita un dato tiene que consultarlo.

Las seis herramientas usadas son: buscar alertas (con filtros), resumen general, historial de consumo, anomalías entre sucursales, pedido por proveedor y problemas de calidad de datos.

Detalles de robustez: cuenta con un system prompt que lo hace responder saludos con naturalidad y ser honesto cuando algo no se puede responder con las herramientas; recorte del historial para no gastar tokens de más; manejo claro del límite de tasa; y el modo simplificado por reglas como red de seguridad si no hay clave.

### 7. Carga de órdenes e historial de semanas

**Carga desde archivo:** en el Simulador se puede subir un CSV con las órdenes de la semana (`sucursal`, `ingrediente_id`, `cantidad_formatos`) y ver las alertas recalcularse. Para que el formato sea inequívoco, hay un botón que descarga una **plantilla** con las órdenes actuales y la validación avisa con claridad si faltan columnas o hay cantidades no numéricas.

**Historial y comparación:** como los datos traen una sola semana de órdenes (y 6 de *consumo*), no se pueden reconstruir alertas de semanas pasadas. Por eso el historial se construye **a medida que se usa la herramienta** de modo que cada semana la gerente guarda una "foto" (snapshot) de las alertas que persiste en un archivo JSON. Con eso el dashboard muestra la evolución en el tiempo y compara la semana actual contra cualquier anterior, listando qué se **resolvió**, qué **empeoró** y qué **mejoró**. Encaja exactamente con el flujo real de "cargar cada semana".

---

## Estructura del proyecto

```
Dashboard Barrio Pizza V1/
├── app.py                      # UI de Streamlit: dashboard, chat, simulador, historial
├── assets/
│   └── logo barrio pizza.jpg
├── data/                       # Los 4 CSV de entrada
│   ├── consumo_historico.csv   # 6 semanas de consumo por sucursal/ingrediente (unidad base)
│   ├── ingredientes.csv        # Maestro: unidad base, formato, factor, proveedor, perecedero
│   ├── inventario_actual.csv   # Stock actual por sucursal/ingrediente (unidad base)
│   └── orden_compra_semana.csv # Órdenes de la semana (en formatos)
├── modules/
│   ├── data_loader.py          # Carga y normaliza los CSV; expone la API de datos
│   ├── forecasting.py          # Proyección robusta (MAD + Theil-Sen + backtest)
│   ├── alerts.py               # Motor de alertas y priorización
│   ├── anomaly.py              # Anomalías entre sucursales (2 enfoques)
│   ├── insights.py             # Pedido corregido y texto por proveedor
│   ├── chat_engine.py          # Chat con IA vía tool calling
│   └── history.py              # Historial de alertas y comparación entre semanas
├── .env.example                # Plantilla de variables de entorno
├── .gitignore
└── requirements.txt
```

Un punto de arquitectura a propósito es que **`data_loader.py` es la única frontera de entrada/salida**. Forecasting, alertas y anomalías operan sobre DataFrames, sin saber de dónde vienen. Eso hace trivial cambiar la fuente de datos (ver la sección de Odoo).

> El historial se guarda en `data/historial_alertas.json` que se genera al usar la app y está en el `.gitignore` (es data, no código).

---

## Limitaciones conocidas

Prefiero ser explícito sobre los bordes del enfoque:

- **Theil-Sen extrapola sin techo.** Con pocos puntos tras excluir atípicas, una pendiente fuerte puede sobredisparar la proyección de una semana. La confianza lo señala, pero no hay un tope duro.
- **Muestra chica (6 semanas, 4 sucursales).** Los umbrales están calibrados de forma conservadora; con más historia se podrían usar pruebas estadísticas formales en vez de heurísticas.
- **Umbrales de anomalía heurísticos** (10% en ratio, 1.3x en cobertura). Son un punto de partida razonable, pero conviene ajustarlos contra datos reales.
- **La proyección asume que la próxima semana se parece a las recientes.** No modela promociones, feriados ni estacionalidad eso requeriría datos externos (calendario, campañas).
- **El historial es local.** Se persiste en un archivo JSON en disco; si se despliega en un entorno efímero (como Streamlit Community Cloud), los snapshots se reinician cuando la app se reinicia.
- **El proxy `cajas_pizza`** asume que refleja bien el volumen real de ventas de cada sucursal.

---

## Cómo lo llevaría a producción con Odoo

Hoy la fuente son CSV pero el diseño está pensado para enchufarse a un ERP. En Odoo los datos ya viven en modelos estándar:

| Dato en el dashboard | Modelo en Odoo |
|---|---|
| Ingredientes / insumos | `product.product` / `product.template` |
| **Conversión base ↔ formato** | `uom.uom` (Unidad de Medida y UoM de compra) |
| Inventario por sucursal | `stock.quant` por `stock.warehouse` / `stock.location` |
| Consumo histórico | `stock.move` (salidas/consumos) o ventas de `pos.order.line` |
| Órdenes de compra | `purchase.order` / `purchase.order.line` |
| Proveedores | `res.partner` + `product.supplierinfo` |

**El plan concreto seria el siguiente:**

1. **Reemplazar `data_loader.py` por un conector de Odoo** para que lea esos modelos vía la External API (XML-RPC/JSON-RPC, o la librería `odoorpc`). El resto del pipeline (proyección, alertas, anomalías) **no cambia** porque trabaja sobre DataFrames y esa separación fue una decisión de diseño justamente para esto.

2. **La conversión de unidades la resuelve Odoo de forma nativa.** Lo que hoy hago a mano con `unidad_base_por_formato` es exactamente el factor entre la UoM base y la UoM de compra de cada producto ya que en producción se lee de `uom.uom` en vez de una columna de CSV.

3. **Cerrar el ciclo (write-back).** En lugar de solo mostrar alertas, empujar el pedido corregido de vuelta como borradores de `purchase.order`. Y como Odoo ya separa las órdenes de compra por proveedor, la vista "pedido corregido por proveedor" mapea 1:1 con esos borradores.

4. **Automatización.** Correr la revisión como un job programado (cron de Odoo) que marca anomalías y le avisa a la gerente o embeber el dashboard como un módulo dentro del propio ERP.

5. **Seguridad.** Una cuenta de servicio con API key y permisos mínimos sobre los modelos relevantes y los secretos por variables de entorno, como ya se hace acá.

---

## Con más tiempo yo agregaría

- **Estacionalidad y calendario de feriados/promociones** en la proyección.
- **Persistencia del historial en una base de datos** en vez de JSON para que sobreviva en entornos desplegados y soporte varios usuarios.
- **Notificaciones automáticas** email/Slack cuando aparecen alertas críticas.
- **Vistas por rol** cada encargado ve su sucursal; la gerente ve todo.

## Proyecto Realizado por: Gonzalo Hooker 