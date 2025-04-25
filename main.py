# Preparando el entorno
import pandas as pd
import numpy as np
import sqlite3 as s
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine, Column, Integer, Float, Date
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, session, Session
import os
import warnings
import datetime
import time
from streamlit_option_menu import option_menu

warnings.filterwarnings("ignore")

st.set_page_config(page_title="EcoForecast", 
                   page_icon="♻️",
                  layout="wide")

st.markdown(
    """
    <style>
        * {
            font-family: "Times New Roman", Times, serif;
        }
        .stTitle, .stHeader {
            font-family: "Times New Roman", Times, serif !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Configuración de la base de datos
DATABASE_URL = "sqlite:///./dataset.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Ruta de la base de datos
dataset = "dataset.db"

# Definir el modelo de datos
class MaterialRecord(Base):
    __tablename__ = "materiales"
    fecha = Column(Date, nullable=False, primary_key=True)
    plastico = Column(Float, default=0.0)
    madera = Column(Float, default=0.0)
    vidrio = Column(Float, default=0.0)
    sargazo = Column(Float, default=0.0)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def show_inicio():
    st.title('Bienvenidos a  EcoForecast ♻️')
    st.header('Herramienta de Predicción de Demanda para Materiales Reciclados')
    st.write('Optimiza tu estrategia de inventario con análisis predictivo avanzado y visualización de datos.')

    # Manejo de navegación
    if "page" in st.session_state:
        if st.session_state["page"] == "Dashboard":
            show_dashboard()
        elif st.session_state["page"] == "Registro de Materiales":
            show_registro_materiales()

    st.markdown("---")

    # Colores
    AZUL_ITLA = "#023877"
    VERDE = "#2D572C"
    FONDO_TARJETA = "#F8F9FA"

    st.subheader("Características Principales")
    st.markdown("Nuestro sistema ofrece herramientas avanzadas para la predicción y análisis de demanda de materiales reciclados.")

    # Contenido de las tarjetas
    caracteristicas = [
        {
            "titulo": "📈 Análisis Predictivo",
            "detalles": [
                "Algoritmos avanzados para predecir la demanda futura de materiales reciclados.",
                "Utiliza datos históricos para generar predicciones precisas."
            ]
        },
        {
            "titulo": "📊 Visualización de Datos",
            "detalles": [
                "Gráficos interactivos y paneles personalizables para analizar tendencias.",
                "Visualiza datos complejos de forma clara y accesible."
            ]
        },
        {
            "titulo": "📂 Registro de Materiales",
            "detalles": [
                "Guarde la cantidad de materiales recolectados durante el tiempo.", 
                "Ingresa la información de los materiales para un análisis personalizado."
            ]
        }
    ]

    # Diseño de las tarjetas
    def mostrar_tarjeta(titulo, detalles):
        with st.container():
            st.markdown(
                f"""
                <div style="background-color:{FONDO_TARJETA}; padding:20px; border-radius:15px; box-shadow: 2px 2px 10px rgba(0,0,0,0.1);">
                    <h4 style="color:{AZUL_ITLA};">{titulo}</h4>
                    <ul>
                        {''.join([f'<li style="color:{VERDE};">{detalle}</li>' for detalle in detalles])}
                    </ul>
                </div>
                """, unsafe_allow_html=True
            )

    # Mostrar tarjetas en una cuadrícula adaptable
    col1, col2, col3 = st.columns([1,1,1])

    with col1:
        mostrar_tarjeta(caracteristicas[0]["titulo"], caracteristicas[0]["detalles"])
    with col2:
        mostrar_tarjeta(caracteristicas[1]["titulo"], caracteristicas[1]["detalles"])
    with col3:
        mostrar_tarjeta(caracteristicas[2]["titulo"], caracteristicas[2]["detalles"])

    st.markdown("---")

    # Contenedor principal
    with st.container():
        st.markdown('<div class="eco-section">', unsafe_allow_html=True)

        # Columnas
        col1 = st.columns(1)[0] 

        with col1:
            st.markdown(
                """
                ### Caso de Estudio: EcoTrofeos

                EcoTrofeos, pioneros en la creación de trofeos con materiales reciclados, lidera la sostenibilidad en su sector. Con **EcoForecast**, la empresa podrá:

                🔹 **Anticipar la Demanda:** Identificar variaciones en la necesidad de materiales reciclados.  
                🔹 **Optimizar la Logística:** Mejorar la recolección y gestión de residuos con análisis predictivo.  
                🔹 **Tomar Decisiones Estratégicas:** Transformar datos en insights en tiempo real.  
                🔹 **Maximizar la Eficiencia:** Digitalizar y optimizar su cadena de suministro.
                """,
                unsafe_allow_html=True
            )

        st.markdown('</div>', unsafe_allow_html=True)

    # Definir el estado de la página seleccionada
    if "pagina" not in st.session_state:
        st.session_state.pagina = "Materiales"

    # Estilo de los botones
    st.markdown("""
        <style>
        .stButton>button {
            width: 200px;
            height: 40px;
            font-size: 16px;
            font-weight: bold;
            margin: 5px;
            border-radius: 8px;
        }
        </style>
    """, unsafe_allow_html=True)

# Importación de datos

# Conexión con la base de datos
conn = s.connect(dataset)
    
# Importación de los datos
df = pd.read_sql("SELECT * FROM materiales;", conn)
    
# Cerrar conexión con la base de datos
conn.commit()
conn.close()
    
# Dataframe largo para los gráficos de barra y de búrbuja
df_long = df.copy()
    
# Index adecuado para el melt
df_long.reset_index(inplace=True)
    
# Eliminar fechas ya que no son necesarias para las barras y no las lee bien
df_long.drop("Fecha", axis=1, inplace=True)

# Cambiar el nombre de las variables a unos ortográficos
df_long.rename(columns={
    "plastico": "Plástico",
    "madera": "Madera",
    "vidrio": "Vidrio",
    "sargazo": "Sargazo"
}, inplace=True)
    
# Transformación a dataset largo
df_long = df_long.melt(id_vars=["index"], var_name="Materiales", value_name="Recolección")

# Cambiar tipo de datos "Fecha" de 'object' a 'datetime'
df["Fecha"] = pd.to_datetime(df["Fecha"])
    
# Transformar la variable a índice
df.set_index("Fecha", inplace=True)

def show_dashboard():

    AZUL_ITLA = "#023877"
    VERDE = "#2D572C"
    
    # Mostrar gráficos en Streamlit
    st.title("📊 Dashboard Analítico")

    # Título y barra lateral en Streamlit
    st.sidebar.title("Filtros del Dashboard")

    # Filtrar por año
    year = st.sidebar.selectbox(
    "Seleccione el año:",
    options=sorted(df.index.year.unique()),
    index=len(df.index.year.unique()) - 1
    )

    # Filtrar datos por el año seleccionado
    filter = df.copy()
    filter = filter[filter.index.year == year]

    st.subheader("Análisis de Materiales Reciclados")
    st.write("Visualiza las tendencias y patrones de recolección de materiales reciclados para la toma de decisiones estrategias sostenibles.")

    st.markdown("---")

    # Agregar tarjetones con métricas relevantes
    col1, col2, col3 = st.columns(3)

    # Tarjetón 1: Recolección total filtrada
    produccion_total = df["plastico"].sum() + df["madera"].sum() + df["vidrio"].sum() + df["sargazo"].sum()
    col1.metric("📦 Recolección Total", f"{produccion_total:.2f} kg")

    # Tarjetón 2: Promedio de recolección filtrado
    promedio_produccion = filter.mean().sum()
    col2.metric(f"📊 Promedio de Recolección ({year})", f"{promedio_produccion:.2f} kg")

    # Tarjetón 3: Recolección de los últimos 12 meses filtrada
    produccion_ultimos_12 = filter.tail(12).sum().sum()
    col3.metric("📅 Últimos 12 Meses", f"{produccion_ultimos_12:.2f} kg")
    
    # Agregar gráfico de barra
    fig_bar = px.bar(
        df_long, 
        x="Materiales", 
        y="Recolección", 
        labels={"Recolección": "Recolección (kg)", "variable": "Materiales"}, 
        barmode="group",
        color_discrete_sequence=["#1C3D5A", "#2A5D6D", "#A8A7A0", "#86A786"]
    )

    # Agregar un título y subtítulo al gráfico de barras
    fig_bar.update_layout(title=go.layout.Title(text="<br>Cantidad Total de Materiales Reciclados <br>por Tipo"))

    # Agregar gráfico de línea
    fig_line = px.line(
        filter, 
        x=filter.index, 
        y=["plastico", "madera", "vidrio", "sargazo"], 
        labels={"value": "Recolección (kg)"}, 
        line_shape="spline", 
        markers=True
    )
       
    # Cambiar el título de la leyenda, el fondo del gráfico de línea y el título
    fig_line.update_layout(legend_title="Materiales", title=go.layout.Title(
      text=f"<br>Evolución de Recolección de Materiales <br>Reciclables en kg ({year})"
    ))

    # Calcular la recolección total por material
    production_totals = filter[['vidrio', 'madera', 'plastico', 'sargazo']].sum()

    # Crear gráfico de pastel
    fig_pie = px.pie(
        production_totals,
        values=production_totals,
        names=production_totals.index,
        color_discrete_sequence=["#1C3D5A", "#2A5D6D", "#A8A7A0", "#86A786"]
    )

    # Agregar un título y subtítulo al gráfico de pastel
    fig_pie.update_layout(title=go.layout.Title(text=f"Proporción de Materiales Recolectados <br>por Tipo ({year})"))

    # Agrupar el dataset por año para el gráfico de columnas apilado
    col = df.groupby(df.index.strftime("%Y"))[["plastico", "madera", "vidrio", "sargazo"]].sum().iloc[-5:]

    # Agregar gráfico de columnas apilado
    fig_col = px.bar(
        col, 
        x=col.index, 
        y=["plastico", "madera", "vidrio", "sargazo"], 
        labels={
            "value": "Recolección (kg)", 
            "variable": "Materiales", 
            "Fecha": "Año de recolección"
        }, 
        barmode="stack",
        color_discrete_sequence=["#1C3D5A", "#2A5D6D", "#A8A7A0", "#86A786"]
    )

    # Agregar un título y subtítulo al gráfico de columnas apiladas
    fig_col.update_layout(title=go.layout.Title(text="Tendencia Anual de Recolección de <br>Materiales Reciclables (Últimos 5 Años)"))

    # Cambiar los nombres de los materiales de la leyenda de 2 gráficos
    for graph in [fig_line, fig_col]:
        graph.for_each_trace(
            lambda trace: trace.update(
                name=trace.name.replace(
                    "plastico", "Plástico"
                ).replace(
                    "madera", "Madera"
                ).replace(
                    "vidrio", "Vidrio"
                ).replace(
                    "sargazo", "Sargazo"
                )
            )
        )

    # Cambiar el color de los labels
    for graph in [fig_bar, fig_line, fig_col, fig_pie]:
      graph.update_layout(xaxis=dict(title_font=dict(color="black", family="Arial Black")), yaxis=dict(title_font=dict(color="black", family="Arial Black")))

    # Colocar y organizar los gráficos en la página
    st.plotly_chart(fig_pie, use_container_width=True)
    st.plotly_chart(fig_bar, use_container_width=True)
    st.plotly_chart(fig_line, use_container_width=True)
    st.plotly_chart(fig_col, use_container_width=True)
    st.subheader("Datos de recolección de los últimos 12 meses")
    st.dataframe(df.tail(12).rename(columns={
        "plastico": "Plástico (kg)",
        "madera": "Madera (kg)",
        "vidrio": "Vidrio (kg)",
        "sargazo": "Sargazo (kg)"
    }))

    
# Definir los colores en CSS
st.markdown("""
    <style>
        /* Cambiar color del selector de fechas */
        .stDateInput input {
            background-color: #2D572C;
            color: white;
            border: 1px solid #023877;
        }
        /* Cambiar color del selector multiselect */
        .stMultiSelect div[role="listbox"] {
            background-color: #2D572C;
            color: white;
            border: 1px solid #023877;
        }
        .stMultiSelect input {
            background-color: #2D572C;
            color: white;
            border: 1px solid #023877;
        }
        .stMultiSelect__option--selected {
            background-color: #023877;
            color: white;
        }
        /* Cambiar color del slider */
        .stSlider .stSlider__range {
            background-color: #023877;
        }
        .stSlider .stSlider__handle {
            background-color: #2D572C;
            border: 2px solid #023877;
        }
        /* Cambiar color de los botones */
        .stButton button {
            background-color: #2D572C;
            color: white;
            border: 1px solid #023877;
        }
        .stButton button:hover {
            background-color: #023877;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

def show_predictor_demanda():
    # Título y barra lateral en Streamlit
    st.title('📈 Predicción de Demanda de Materiales')
    st.sidebar.title("Selector del Predictor de Demanda")

    # Agregar un filtro de intervalo de fechas
    dates = [
        pd.to_datetime(d) for d in st.sidebar.date_input(
            "Seleccione el período deseado:", 
            [df.index[-12], df.index.max()], 
            min_value=df.index.min(),
            max_value=df.index.max()
        )
    ]

    # Detener streamlit hasta que se hayan seleccionado ambas fechas
    if len(dates) != 2:
        st.sidebar.info("Seleccione el rango completo del período")
        st.stop()

    # Colocar los límites de fecha
    date_begin, date_end = dates

    # Modificador del tamaño de las predicciones (por meses)
    meter = st.sidebar.slider("¿Por cuantos meses desea predecir?", 3, 12, 12)

    # Botón para ejecutar predicción
    execute = st.sidebar.button("Ejecutar Predicción")

    if execute:

        # Variable total que servirá para predecir demanda
        total = np.array(df["plastico"] + df["madera"] + df["vidrio"] + df["sargazo"])
    
        # Integración de las fechas al total de materiales (por kg)
        total = pd.Series(data=total, index=df.index)

        total = total[(total.index >= date_begin) & (total.index <= date_end)]
    
        # Obtención de todos los valores de los parámetros
        p = q = range(0, 6)
        d = range(0, 3)
    
        # Declaración de todos los valores de los parámetros
        pdq = [(x, y, z) for x in p for y in d for z in q]
    
        # Método de Grid Search para escoger los mejores parámetros
        best_aic = np.inf
        best_params = None

        with st.spinner("Ejecutando predicciones"):
            for param in pdq:
                try:
                    model = ARIMA(total, order=param)
                    model_fit = model.fit()
                    if model_fit.aic < best_aic:
                        best_aic = model_fit.aic
                        best_params = param
                except:
                    continue
    
        # Creación del modelo ARIMA con los mejores parámetros
        model = ARIMA(total, order=best_params)
        model_fit = model.fit()
    
        # Predicciones a través del medidor
        predictions = model_fit.forecast(steps=meter)
    
        # Intervalo de confianza para conocer límites
        confidence = model_fit.get_forecast(steps=meter)
        conf_int = confidence.conf_int()
    
        # Agregar gráfico de línea
        fig_line = go.Figure()

        fig_line.add_trace(go.Scatter(
            x=total.index, y=total.values,
            mode='lines',
            name='Recolección (kg)',
            line=dict(color='#1C3D5A')
        ))
        
        fig_line.add_trace(go.Scatter(
            x=predictions.index, y=predictions.values,
            mode='lines',
            name='Predicciones',
            line=dict(color='#86A786')
        ))
        
        fig_line.add_trace(go.Scatter(
            x=predictions.index.tolist() + predictions.index[::-1].tolist(),
            y=conf_int.iloc[:, 1].tolist() + conf_int.iloc[:, 0][::-1].tolist(),
            fill='toself',
            fillcolor='rgba(168,167,160,0.2)',  # color con opacidad
            line=dict(color='rgba(168,167,160,0)'),  # línea invisible
            name='Intervalo de Confianza'
        ))
        
        fig_line.update_layout(
            title=f"Recolección y predicciones de materiales en kg (histórico total con prox. {meter} meses de predicción)",
            xaxis_title="Fecha",
            yaxis_title="Recolección kg",
            legend_title="Series",
            template="plotly_white"
        )

        # Imprimiendo el gráfico
        st.plotly_chart(fig_line, use_container_width=True)

    else:
        # Instrucción para que el usuario presione el botón
        st.info('Presione el botón  "Ejecutar predicción" para ejecutar predicción')

def show_registro_materiales():
    st.title('🗃️ Registro de Materiales Recolectados')
    st.subheader('Ingrese la cantidad de materiales recolectados en kilógramos para su análisis y predicción de demanda.')

    # Seleccionador de nuevas fechas
    start_date = df.index.max()
    end_date = pd.to_datetime(datetime.date.today())

    # Filtrar solamente las fechas del primer día hábil
    month_dates = pd.date_range(start=start_date, end=end_date, freq="BMS")

    # Filtrar solamente las fechas no existentes dentro del dataframe
    save_dates = [month.strftime("%Y-%m-%d") for month in month_dates if month not in df.index]

    # Guardar en caso de que no existan más fechas
    if save_dates:
        
        # Entradas de los materiales
        fecha = st.selectbox("Fecha de Registro (Primer día hábil del mes):", save_dates)
        plastico = st.number_input('Plástico (kg)', min_value=0.0, step=0.1)
        madera = st.number_input('Madera (kg)', min_value=0.0, step=0.1)
        vidrio = st.number_input('Vidrio (kg)', min_value=0.0, step=0.1)
        sargazo = st.number_input('Sargazo (kg)', min_value=0.0, step=0.1)

    # Mensaje si todas las fechas están ocupadas
    else:
        st.warning("Se registraron todas las fechas existentes")
        st.info(f"Última fecha registrada {end_date.date()}")
        st.stop()
    
    # Botón para guardar
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button('Guardar Registro'):
            try:
                # Obtener la conexión a la base de datos
                db = next(get_db())
                
                # Crear un nuevo registro
                nuevo_registro = MaterialRecord(
                    fecha=pd.to_datetime(fecha).date(),
                    plastico=plastico,
                    madera=madera,
                    vidrio=vidrio,
                    sargazo=sargazo
                )
                
                # Agregar el registro a la sesión
                db.add(nuevo_registro)
                db.commit()  # Confirmar la transacción

                # Verificar si se guardó correctamente
                db.refresh(nuevo_registro)  # Refrescar el objeto para obtener el ID autoincremental

                # Mostrar mensaje de éxito
                st.success('¡Registro guardado exitosamente!')
                
                # Mostrar el objeto como un diccionario para una visualización más clara
                st.write("Detalles del registro guardado:", nuevo_registro.__dict__)

                # Esperar 7 segundos antes de limpiar
                time.sleep(3)

                # Limpiar los campos de entrada (redirigir a la página)
                st.rerun()  # Reiniciar la app para limpiar los campos

            except Exception as e:
                # Si hay un error, revertir los cambios
                db.rollback()
                st.error(f'Error al guardar el registro: {str(e)}')
                st.write(str(e))

# Configuración del menú de navegación
if "pagina" not in st.session_state:
    st.session_state["pagina"] = "Inicio"

# Imagen de la empresa
st.sidebar.image("imagenes/ecoforecast logo-Photoroom (1).png")

with st.sidebar:
    selected = option_menu(
        menu_title=None,
        options=["Inicio", "Dashboard", "Predictor de Demanda", "Registro de Materiales"],
        icons=["house", "bar-chart-line", "graph-up-arrow", "clipboard2-check"],
        default_index=0,
        styles={
            "container": {
                "padding": "0!important", 
                "background-color": "#dae3da",
              # Permite centrar contenido
                "flex-direction": "column",  # Asegura que los elementos estén en columna
                "justify-content": "center",
                "font-weight": "bold",
                "font-family": "Times New Roman, Times, serif"  # Centra verticalmente
                
            },
            "icon": {"color": "black", "font-size": "15px"},
            "nav-link": {
                "font-size": "15px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#eee",
                "font-weight": "bold",  # Fuente normal para los enlaces no seleccionados
            },
            "nav-link-selected": {
                "background-color": "#979c97",
                "font-weight": "bold",  # Mantiene la negrita en la pestaña seleccionada
                "color": "black", # Reduce el grosor de la fuente al seleccionar
            },
            "icon-selected": {
                "color": "black",  # Íconos seleccionados en blanco
            },
        }
    )

    
# Verificar si estamos en la página del dashboard
if selected == "Inicio":
     show_inicio()
elif selected == "Dashboard":
    show_dashboard()
elif selected == "Predictor de Demanda":
    show_predictor_demanda()
elif selected == "Registro de Materiales":
    show_registro_materiales()
