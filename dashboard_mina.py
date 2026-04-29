import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# =========================
# CONFIGURACIÓN
# =========================
st.set_page_config(page_title="Dashboard Minero", layout="wide")

st.title("🚛 Dashboard de Acarreo y Transporte en Mina a Tajo Abierto")
st.caption("Recorrido 2D/3D, KPIs, estados operativos, alertas y detección de posibles choques con radio de seguridad.")

# =========================
# CARGAR DATOS
# =========================
@st.cache_data
def cargar_datos():
    df = pd.read_excel(
        "data/vehicle_positions.xlsx",
        dtype={"Vehiculo": str}
    )

    df["Vehiculo"] = df["Vehiculo"].str.strip()
    df["Tiempo"] = pd.to_datetime(df["Tiempo"], format="%H:%M:%S")
    df = df.sort_values(["Vehiculo", "Tiempo"]).reset_index(drop=True)

    df["dX"] = df.groupby("Vehiculo")["X"].diff()
    df["dY"] = df.groupby("Vehiculo")["Y"].diff()
    df["dZ"] = df.groupby("Vehiculo")["Z"].diff()

    df["Distancia_step"] = np.sqrt(df["dX"]**2 + df["dY"]**2 + df["dZ"]**2).fillna(0)
    df["dt"] = df.groupby("Vehiculo")["Tiempo"].diff().dt.total_seconds().fillna(0)

    return df

df = cargar_datos()

# =========================
# ESTADO OPERATIVO
# =========================
def clasificar_estado(row):
    velocidad = row["Velocidad"]
    llenado = row["Llenado"]
    tolva = str(row["Tolva"]).lower()
    abasteciendo = str(row["Abasteciendo"]).lower()

    if abasteciendo == "si":
        return "Abasteciendo"
    if tolva == "arriba":
        return "Descargando"
    if velocidad > 0 and llenado >= 80:
        return "Transportando cargado"
    if velocidad > 0 and llenado < 80:
        return "Retornando vacío"
    if velocidad == 0 and llenado >= 80:
        return "Detenido cargado / Cola"
    if velocidad == 0 and llenado < 80:
        return "Cargando / Espera"

    return "Operación normal"

df["Estado_Operativo"] = df.apply(clasificar_estado, axis=1)

vehiculos = sorted(df["Vehiculo"].unique())
tiempos = sorted(df["Tiempo"].unique())

# =========================
# SESSION STATE PARA PLAY
# =========================
if "playing" not in st.session_state:
    st.session_state.playing = False

if "tiempo_idx" not in st.session_state:
    st.session_state.tiempo_idx = 0

# =========================
# SIDEBAR
# =========================
st.sidebar.header("⚙️ Controles")

vehiculo = st.sidebar.selectbox("Selecciona un vehículo", vehiculos)
ver_todos = st.sidebar.checkbox("Ver todos los vehículos", value=True)

radio_seguridad = st.sidebar.slider(
    "Radio de seguridad del camión (m)",
    min_value=5,
    max_value=20,
    value=10,
    step=1
)

velocidad_animacion = st.sidebar.slider(
    "Velocidad de animación",
    min_value=0.05,
    max_value=1.00,
    value=0.20,
    step=0.05
)

col_play, col_pause, col_reset = st.sidebar.columns(3)

if col_play.button("▶ Play"):
    st.session_state.playing = True

if col_pause.button("⏸ Pausa"):
    st.session_state.playing = False

if col_reset.button("⏮ Reset"):
    st.session_state.playing = False
    st.session_state.tiempo_idx = 0

st.session_state.tiempo_idx = st.sidebar.slider(
    "Tiempo de simulación",
    min_value=0,
    max_value=len(tiempos) - 1,
    value=st.session_state.tiempo_idx
)

tiempo_actual = tiempos[st.session_state.tiempo_idx]

rpm_alerta = st.sidebar.number_input(
    "RPM alto para alerta",
    min_value=1000,
    max_value=3000,
    value=2000,
    step=100
)

vel_baja_alerta = st.sidebar.number_input(
    "Velocidad baja para alerta",
    min_value=0.0,
    max_value=20.0,
    value=5.0,
    step=1.0
)

# =========================
# FILTROS
# =========================
df_tiempo = df[df["Tiempo"] <= tiempo_actual].copy()
df_instante = df[df["Tiempo"] == tiempo_actual].copy()

if ver_todos:
    df_plot = df_tiempo.copy()
else:
    df_plot = df_tiempo[df_tiempo["Vehiculo"] == vehiculo].copy()

df_vehiculo = df[df["Vehiculo"] == vehiculo].copy()

# =========================
# KPIs
# =========================
st.subheader("📊 KPIs Operativos")

kpis = df.groupby("Vehiculo").agg(
    Velocidad_Max=("Velocidad", "max"),
    Velocidad_Prom=("Velocidad", "mean"),
    Dist_Total=("Distancia_step", "sum"),
    Tiempo_Total=("dt", "sum"),
    RPM_Prom=("RPM", "mean"),
    Combustible_Prom=("Combustible", "mean")
)

idle = df[df["Velocidad"] == 0].groupby("Vehiculo")["dt"].sum()
mov = df[df["Velocidad"] > 0].groupby("Vehiculo")["dt"].sum()
giros = df[df["Giro_Brusco"].astype(str).str.lower() == "si"].groupby("Vehiculo").size()
abast = df[df["Abasteciendo"].astype(str).str.lower() == "si"].groupby("Vehiculo").size()
tolva_arriba = df[df["Tolva"].astype(str).str.lower() == "arriba"].groupby("Vehiculo")["dt"].sum()

kpis["Idle_%"] = (idle / kpis["Tiempo_Total"] * 100).fillna(0)
kpis["Movimiento_%"] = (mov / kpis["Tiempo_Total"] * 100).fillna(0)
kpis["Giros_Bruscos"] = giros.reindex(kpis.index, fill_value=0)
kpis["Abastecimientos"] = abast.reindex(kpis.index, fill_value=0)
kpis["Tolva_Arriba_%"] = (tolva_arriba / kpis["Tiempo_Total"] * 100).fillna(0)

kpi_v = kpis.loc[vehiculo]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Velocidad promedio", f"{kpi_v['Velocidad_Prom']:.2f}")
col2.metric("Velocidad máxima", f"{kpi_v['Velocidad_Max']:.2f}")
col3.metric("Distancia total", f"{kpi_v['Dist_Total']:.2f} m")
col4.metric("RPM promedio", f"{kpi_v['RPM_Prom']:.0f}")

col5, col6, col7, col8 = st.columns(4)
col5.metric("Idle %", f"{kpi_v['Idle_%']:.2f}%")
col6.metric("Movimiento %", f"{kpi_v['Movimiento_%']:.2f}%")
col7.metric("Giros bruscos", int(kpi_v["Giros_Bruscos"]))
col8.metric("Abastecimientos", int(kpi_v["Abastecimientos"]))

col9, col10 = st.columns(2)
col9.metric("Combustible promedio", f"{kpi_v['Combustible_Prom']:.2f}")
col10.metric("Tolva arriba %", f"{kpi_v['Tolva_Arriba_%']:.2f}%")

# =========================
# ESTADO ACTUAL
# =========================
st.subheader("🚦 Estado actual del vehículo seleccionado")

estado_actual = df_instante[df_instante["Vehiculo"] == vehiculo]

if not estado_actual.empty:
    fila = estado_actual.iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tiempo", tiempo_actual.strftime("%H:%M:%S"))
    c2.metric("Vehículo", fila["Vehiculo"])
    c3.metric("Estado", fila["Estado_Operativo"])
    c4.metric("Velocidad", f"{fila['Velocidad']:.2f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Llenado", f"{fila['Llenado']:.0f}%")
    c6.metric("RPM", f"{fila['RPM']:.0f}")
    c7.metric("Tolva", fila["Tolva"])
    c8.metric("Combustible", f"{fila['Combustible']:.2f}")

# =========================
# DISTRIBUCIÓN DE ESTADOS
# =========================
st.subheader("📌 Distribución de estados operativos")

estado_count = df_vehiculo["Estado_Operativo"].value_counts().reset_index()
estado_count.columns = ["Estado", "Cantidad"]

fig_estado = go.Figure()
fig_estado.add_trace(go.Bar(
    x=estado_count["Estado"],
    y=estado_count["Cantidad"],
    text=estado_count["Cantidad"],
    textposition="auto"
))
fig_estado.update_layout(
    height=380,
    xaxis_title="Estado operativo",
    yaxis_title="Cantidad de registros"
)
st.plotly_chart(fig_estado, use_container_width=True)

# =========================
# RECORRIDO 3D
# =========================
st.subheader("🗺️ Reconstrucción del recorrido 3D")

fig3d = go.Figure()

for v in df_plot["Vehiculo"].unique():
    d = df_plot[df_plot["Vehiculo"] == v]

    fig3d.add_trace(go.Scatter3d(
        x=d["X"],
        y=d["Y"],
        z=d["Z"],
        mode="lines",
        name=f"Recorrido {v}",
        line=dict(width=5)
    ))

for _, row in df_instante.iterrows():
    fig3d.add_trace(go.Scatter3d(
        x=[row["X"]],
        y=[row["Y"]],
        z=[row["Z"]],
        mode="markers+text",
        text=[str(row["Vehiculo"])],
        textposition="top center",
        name=f"Actual {row['Vehiculo']}",
        marker=dict(size=7)
    ))

fig3d.update_layout(
    height=650,
    scene=dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z"
    )
)

st.plotly_chart(fig3d, use_container_width=True)

# =========================
# RECORRIDO 2D CON RADIO DE SEGURIDAD
# =========================
st.subheader("📍 Recorrido 2D con radio de seguridad")

fig2d = go.Figure()

for v in df_plot["Vehiculo"].unique():
    d = df_plot[df_plot["Vehiculo"] == v]

    fig2d.add_trace(go.Scatter(
        x=d["X"],
        y=d["Y"],
        mode="lines",
        name=f"Recorrido {v}"
    ))

theta = np.linspace(0, 2 * np.pi, 80)

for _, row in df_instante.iterrows():
    x0 = row["X"]
    y0 = row["Y"]

    x_circulo = x0 + radio_seguridad * np.cos(theta)
    y_circulo = y0 + radio_seguridad * np.sin(theta)

    fig2d.add_trace(go.Scatter(
        x=x_circulo,
        y=y_circulo,
        mode="lines",
        name=f"Radio {row['Vehiculo']}",
        line=dict(dash="dash")
    ))

    fig2d.add_trace(go.Scatter(
        x=[x0],
        y=[y0],
        mode="markers+text",
        text=[str(row["Vehiculo"])],
        textposition="top center",
        name=f"Actual {row['Vehiculo']}",
        marker=dict(size=10)
    ))

fig2d.update_layout(
    height=600,
    xaxis_title="X",
    yaxis_title="Y"
)

st.plotly_chart(fig2d, use_container_width=True)

# =========================
# DETECCIÓN DE CHOQUES POR ESFERAS
# =========================
st.subheader("⚠️ Posibles choques por intersección de radios de seguridad")

choques = []
limite_interseccion = 2 * radio_seguridad

for i in range(len(df_instante)):
    for j in range(i + 1, len(df_instante)):
        v1 = df_instante.iloc[i]
        v2 = df_instante.iloc[j]

        dist = np.sqrt(
            (v1["X"] - v2["X"])**2 +
            (v1["Y"] - v2["Y"])**2 +
            (v1["Z"] - v2["Z"])**2
        )

        if dist <= limite_interseccion:
            choques.append({
                "Tiempo": tiempo_actual.strftime("%H:%M:%S"),
                "Vehiculo_1": v1["Vehiculo"],
                "Vehiculo_2": v2["Vehiculo"],
                "Distancia_centros": round(dist, 2),
                "Límite_intersección": limite_interseccion,
                "Criterio": "Intersección de radios de seguridad"
            })

if choques:
    st.error("⚠️ Posible choque o proximidad peligrosa detectada.")
    st.dataframe(pd.DataFrame(choques), use_container_width=True)
else:
    st.success("✔ Sin intersección de radios de seguridad en este instante.")

# =========================
# ALERTAS OPERATIVAS
# =========================
st.subheader("🚨 Alertas operativas")

alertas = []

for _, row in df_instante[df_instante["Giro_Brusco"].astype(str).str.lower() == "si"].iterrows():
    alertas.append({
        "Tiempo": tiempo_actual.strftime("%H:%M:%S"),
        "Vehículo": row["Vehiculo"],
        "Alerta": "Giro brusco",
        "Detalle": "Maniobra brusca detectada."
    })

for _, row in df_instante[
    (df_instante["RPM"] >= rpm_alerta) &
    (df_instante["Velocidad"] <= vel_baja_alerta)
].iterrows():
    alertas.append({
        "Tiempo": tiempo_actual.strftime("%H:%M:%S"),
        "Vehículo": row["Vehiculo"],
        "Alerta": "RPM alto con velocidad baja",
        "Detalle": "Posible esfuerzo excesivo, subida o ineficiencia."
    })

for _, row in df_instante[df_instante["Abasteciendo"].astype(str).str.lower() == "si"].iterrows():
    alertas.append({
        "Tiempo": tiempo_actual.strftime("%H:%M:%S"),
        "Vehículo": row["Vehiculo"],
        "Alerta": "Abastecimiento",
        "Detalle": "Vehículo en abastecimiento de combustible."
    })

for ch in choques:
    alertas.append({
        "Tiempo": ch["Tiempo"],
        "Vehículo": f"{ch['Vehiculo_1']} - {ch['Vehiculo_2']}",
        "Alerta": "Posible choque",
        "Detalle": f"Distancia entre centros: {ch['Distancia_centros']} m."
    })

if alertas:
    st.warning("Se encontraron alertas en el instante seleccionado.")
    st.dataframe(pd.DataFrame(alertas), use_container_width=True)
else:
    st.success("✔ No hay alertas operativas en este instante.")

# =========================
# GRÁFICOS DEL VEHÍCULO
# =========================
st.subheader("📈 Comportamiento del vehículo seleccionado")

fig_vel = go.Figure()
fig_vel.add_trace(go.Scatter(
    x=df_vehiculo["Tiempo"],
    y=df_vehiculo["Velocidad"],
    mode="lines",
    name="Velocidad"
))
fig_vel.update_layout(height=330, xaxis_title="Tiempo", yaxis_title="Velocidad")
st.plotly_chart(fig_vel, use_container_width=True)

fig_rpm = go.Figure()
fig_rpm.add_trace(go.Scatter(
    x=df_vehiculo["Tiempo"],
    y=df_vehiculo["RPM"],
    mode="lines",
    name="RPM"
))
fig_rpm.update_layout(height=330, xaxis_title="Tiempo", yaxis_title="RPM")
st.plotly_chart(fig_rpm, use_container_width=True)

fig_comb = go.Figure()
fig_comb.add_trace(go.Scatter(
    x=df_vehiculo["Tiempo"],
    y=df_vehiculo["Combustible"],
    mode="lines",
    name="Combustible"
))
fig_comb.update_layout(height=330, xaxis_title="Tiempo", yaxis_title="Combustible")
st.plotly_chart(fig_comb, use_container_width=True)

# =========================
# TABLAS
# =========================
st.subheader("📋 KPIs generales")
st.dataframe(kpis.round(2), use_container_width=True)

st.subheader("📋 Datos filtrados")
st.dataframe(df_plot, use_container_width=True)

# =========================
# PLAY AUTOMÁTICO
# =========================
if st.session_state.playing:
    if st.session_state.tiempo_idx < len(tiempos) - 1:
        time.sleep(velocidad_animacion)
        st.session_state.tiempo_idx += 1
        st.rerun()
    else:
        st.session_state.playing = False
