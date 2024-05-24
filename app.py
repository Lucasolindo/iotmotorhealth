import streamlit as st
import webbrowser
import psycopg2
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from keras.models import load_model
from datetime import datetime
from scipy.fftpack import fft
from scipy.signal import find_peaks

# Configuração do banco de dados
DB_HOST = 'localhost'
DB_NAME = 'iotmotorhealth'
DB_USER = 'postgres'
DB_PASSWORD = '123456'

# Função para conectar ao banco de dados
def get_connection():
    conn = psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST
    )
    return conn

# Função para obter os dados do banco de dados
@st.cache_data(ttl=60)
def get_data():
    conn = get_connection()
    query = "SELECT * FROM iotmotorhealth ORDER BY timestamp DESC LIMIT 100"
    data = pd.read_sql(query, conn)
    conn.close()
    return data

# Função para criar gráficos de aceleração
def plot_acceleration(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['timestamp'], y=data['aceleracao_x'], mode='lines', name='Aceleração X'))
    fig.add_trace(go.Scatter(x=data['timestamp'], y=data['aceleracao_y'], mode='lines', name='Aceleração Y'))
    fig.add_trace(go.Scatter(x=data['timestamp'], y=data['aceleracao_z'], mode='lines', name='Aceleração Z'))
    
    # Adicionando linhas de limite
    limit_min = -1.5
    limit_max = 11
    fig.add_shape(type="line", x0=data['timestamp'].min(), x1=data['timestamp'].max(), y0=limit_min, y1=limit_min,
                  line=dict(color="Red", width=2, dash="dash"), name="Limite Mínimo")
    fig.add_shape(type="line", x0=data['timestamp'].min(), x1=data['timestamp'].max(), y0=limit_max, y1=limit_max,
                  line=dict(color="Green", width=2, dash="dash"), name="Limite Máximo")
    
    fig.update_layout(title='Gráficos de Aceleração', xaxis_title='Timestamp', yaxis_title='Vibração (mm/s)')
    return fig

# Função para criar gráficos de velocidade
def plot_velocity(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['timestamp'], y=data['velocidade_x'], mode='lines', name='Velocidade X'))
    fig.add_trace(go.Scatter(x=data['timestamp'], y=data['velocidade_y'], mode='lines', name='Velocidade Y'))
    fig.add_trace(go.Scatter(x=data['timestamp'], y=data['velocidade_z'], mode='lines', name='Velocidade Z'))
    
    # Adicionando linhas de limite
    limit_min = -2.5
    limit_max = 2.5
    fig.add_shape(type="line", x0=data['timestamp'].min(), x1=data['timestamp'].max(), y0=limit_min, y1=limit_min,
                  line=dict(color="Red", width=2, dash="dash"), name="Limite Mínimo")
    fig.add_shape(type="line", x0=data['timestamp'].min(), x1=data['timestamp'].max(), y0=limit_max, y1=limit_max,
                  line=dict(color="Green", width=2, dash="dash"), name="Limite Máximo")
    
    fig.update_layout(title='Gráficos de Velocidade', xaxis_title='Timestamp', yaxis_title='Vibração (mm/s)')
    return fig

# Função para criar gráficos de temperatura
def plot_temperature(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['timestamp'], y=data['temperatura'], mode='lines', name='Temperatura', line=dict(color='red')))
    
    # Adicionando linhas de limite
    limit_min = 20.0
    limit_max = 30.0
    fig.add_shape(type="line", x0=data['timestamp'].min(), x1=data['timestamp'].max(), y0=limit_min, y1=limit_min,
                  line=dict(color="Red", width=2, dash="dash"), name="Limite Mínimo")
    fig.add_shape(type="line", x0=data['timestamp'].min(), x1=data['timestamp'].max(), y0=limit_max, y1=limit_max,
                  line=dict(color="Green", width=2, dash="dash"), name="Limite Máximo")
    
    fig.update_layout(title='Gráfico de Temperatura', xaxis_title='Timestamp', yaxis_title='Temperatura')
    return fig

# Função para criar gauges de aceleração
def create_acceleration_gauge(data, axis):
    return go.Figure(go.Indicator(
        mode="gauge+number",
        value=data[f'aceleracao_{axis}'].iloc[-1],
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Aceleração {axis.upper()}"},
        gauge={'axis': {'range': [-10, 10]}}
    )).update_layout(height=250, margin={'t': 0, 'b': 0})

# Função para criar gauges de velocidade
def create_velocity_gauge(data, axis):
    return go.Figure(go.Indicator(
        mode="gauge+number",
        value=data[f'velocidade_{axis}'].iloc[-1],
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Velocidade {axis.upper()}"},
        gauge={'axis': {'range': [-180, 180]}}
    )).update_layout(height=250, margin={'t': 0, 'b': 0})

def create_speedometer(data):
    # Cálculo da magnitude da velocidade
    speed = np.sqrt(data['aceleracao_x']**2 + data['aceleracao_y']**2 + data['aceleracao_z']**2).iloc[-1]
    
    return go.Figure(go.Indicator(
        mode = "gauge+number",
        value = speed,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Velocidade Atual"},
        gauge = {'axis': {'range': [None, 100]},
                 'steps' : [
                     {'range': [0, 25], 'color': "lightgray"},
                     {'range': [25, 75], 'color': "gray"},
                     {'range': [75, 100], 'color': "red"}],
                 'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 90}})).update_layout(height=250, margin={'t': 0, 'b': 0})

# Função para criar gráfico de barras das previsões
def create_predictions_bar_chart(predictions):
    df = pd.DataFrame(predictions, columns=['Previsão'])
    fig = px.bar(df, y='Previsão', title='Previsões do Modelo', labels={'index': 'Índice', 'Previsão': 'Valor da Previsão'})
    return fig

# Função para criar gráfico de linhas das classificações com limitadores
def create_classification_line_chart(data):
    # Criar traces para cada classificação
    traces = []
    
    # Bom
    data_bom = data[data['classification'] == 'BOM']
    traces.append(go.Scatter(x=data_bom['timestamp'], y=data_bom['predictions'], mode='lines', name='BOM', line=dict(color='green')))
    
    # Adequado
    data_adequado = data[data['classification'] == 'ADEQUADO']
    traces.append(go.Scatter(x=data_adequado['timestamp'], y=data_adequado['predictions'], mode='lines', name='ADEQUADO', line=dict(color='blue')))
    
    # Admissível
    data_admissivel = data[data['classification'] == 'ADMISSÍVEL']
    traces.append(go.Scatter(x=data_admissivel['timestamp'], y=data_admissivel['predictions'], mode='lines', name='ADMISSÍVEL', line=dict(color='orange')))
    
    # Inadmissível
    data_inadmissivel = data[data['classification'] == 'INADMISSÍVEL']
    traces.append(go.Scatter(x=data_inadmissivel['timestamp'], y=data_inadmissivel['predictions'], mode='lines', name='INADMISSÍVEL', line=dict(color='red')))
    
    fig = go.Figure(traces)
    
    # Adicionar shapes para as áreas de classificação
    fig.add_shape(type="rect", x0=data['timestamp'].min(), x1=data['timestamp'].max(), y0=-0.28, y1=0.28,
                  fillcolor="green", opacity=0.2, layer="below", line_width=0, name="BOM")
    fig.add_shape(type="rect", x0=data['timestamp'].min(), x1=data['timestamp'].max(), y0=-0.45, y1=0.45,
                  fillcolor="blue", opacity=0.2, layer="below", line_width=0, name="ADEQUADO")
    fig.add_shape(type="rect", x0=data['timestamp'].min(), x1=data['timestamp'].max(), y0=-0.71, y1=0.71,
                  fillcolor="orange", opacity=0.2, layer="below", line_width=0, name="ADMISSÍVEL")
    fig.add_shape(type="rect", x0=data['timestamp'].min(), x1=data['timestamp'].max(), y0=-1.12, y1=1.12,
                  fillcolor="red", opacity=0.2, layer="below", line_width=0, name="INADMISSÍVEL")
    
    fig.update_layout(title='Classificação da Vibração ao Longo do Tempo', xaxis_title='Timestamp', yaxis_title='Valor da Previsão')
    return fig

# Função para classificar o nível de vibração conforme a norma ISO 10816 ou ISO 2372
def classify_vibration(value, power):
    if power <= 20:
        if -0.28 <= value <= 0.28:
            return "BOM"
        elif -0.45 <= value <= 0.45:
            return "ADEQUADO"
        elif -0.71 <= value <= 0.71:
            return "ADEQUADO"
        elif -1.12 <= value <= 1.12:
            return "ADMISSÍVEL"
        elif -1.8 <= value <= 1.8:
            return "ADMISSÍVEL"
        else:
            return "INADMISSÍVEL"
    elif power <= 100:
        if -0.45 <= value <= 0.45:
            return "BOM"
        elif -0.71 <= value <= 0.71:
            return "ADEQUADO"
        elif -1.12 <= value <= 1.12:
            return "ADMISSÍVEL"
        elif -1.8 <= value <= 1.8:
            return "ADMISSÍVEL"
        elif -2.8 <= value <= 2.8:
            return "ADMISSÍVEL"
        else:
            return "INADMISSÍVEL"
    else:
        if -0.71 <= value <= 0.71:
            return "BOM"
        elif -1.12 <= value <= 1.12:
            return "ADEQUADO"
        elif -1.8 <= value <= 1.8:
            return "ADEQUADO"
        elif -2.8 <= value <= 2.8:
            return "ADMISSÍVEL"
        elif -4.5 <= value <= 4.5:
            return "ADMISSÍVEL"
        else:
            return "INADMISSÍVEL"

# Função para identificar picos relevantes nos espectros de frequência
def identify_peaks(xf, yf, height=0.1):
    peaks, _ = find_peaks(yf, height=height)
    return xf[peaks], yf[peaks]

# Função para classificar o estado da máquina com base nos picos de frequência
def classify_spectrum(peaks_x, peaks_y):
    classifications = {
        "Máquina em boas condições": False,
        "Desbalanceamento": False,
        "Desalinhamento": False,
        "Eixo flexionado": False,
        "Desgaste nos mancais": False,
        "Falta de oil whirl": False,
        "Excentricidade": False,
        "Atrito no motor": False,
        "Engrenagem em ótimo estado": False,
        "Engrenagem em estado ruim": False,
        "Barras do rotor quebradas": False,
        "Excentricidade do estator": False,
        "Rolamento - CPB": False,
        "Defeitos no aro interior do rolamento": False,
        "Defeitos no aro exterior do rolamento": False
    }

    # Padrões de frequência para cada condição
    patterns = {
        "Desbalanceamento": [1],
        "Desalinhamento": [1, 2, 3],
        "Eixo flexionado": [1, 2],
        "Desgaste nos mancais": [1, 2, 3],
        "Falta de oil whirl": [0.42, 0.48],
        "Excentricidade": [1, 2],
        "Atrito no motor": [0.5, 1, 1.5, 2, 2.5],
        "Engrenagem em estado ruim": [2, 3, 4, 5],
        "Barras do rotor quebradas": [n for n in range(1, 11)],
        "Excentricidade do estator": [2, 4, 6, 8, 10],
        "Defeitos no aro interior do rolamento": [2, 4, 6, 8, 10],
        "Defeitos no aro exterior do rolamento": [2, 4, 6, 8, 10]
    }

    for condition, freqs in patterns.items():
        for freq in freqs:
            if np.any(np.isclose(peaks_x, freq, atol=0.1)):
                classifications[condition] = True

    if not any(classifications.values()):
        classifications["Máquina em boas condições"] = True

    return classifications

# Função para calcular a FFT
def apply_fft(data, sampling_rate):
    data = data.to_numpy()  # Converte a série Pandas para um array NumPy
    n = len(data)
    T = 1.0 / sampling_rate
    yf = fft(data)
    xf = np.linspace(0.0, 1.0/(2.0*T), n//2)
    return xf, 2.0/n * np.abs(yf[:n//2])

# Carregar o modelo treinado
model = load_model('motor_health_model.keras')

# Configuração inicial da página do Streamlit
st.markdown("<h1 style='text-align: center;'>Dashboard de Monitoramento - Motor Health</h1>", unsafe_allow_html=True)
st.sidebar.title("Configurações")

# Adicionar botão para redirecionar para o Dashboard de Tempo Real
if st.button('Dashboard'):
    webbrowser.open_new_tab('http://127.0.0.1:1880/ui/#!/0?socketid=VqMiP-PBDjshc-MfAAAB')

# Botão para carregar os dados
if st.sidebar.button('Carregar Dados'):
    data = get_data()
    
    st.write("Últimos 100 registros de dados:")
    st.write(data)
    
    st.write("Gráficos de Acompanhamento")
    fig_acceleration = plot_acceleration(data)
    st.plotly_chart(fig_acceleration)

    fig_velocity = plot_velocity(data)
    st.plotly_chart(fig_velocity)

    fig_temperature = plot_temperature(data)
    st.plotly_chart(fig_temperature)

    st.write("Gauges de Monitoramento")
    
    st.subheader("Velocímetro")
    speedometer = create_speedometer(data)
    st.plotly_chart(speedometer)

    st.subheader("Aceleração X")
    accel_gauge_x = create_acceleration_gauge(data, 'x')
    st.plotly_chart(accel_gauge_x)

    st.subheader("Aceleração Y")
    accel_gauge_y = create_acceleration_gauge(data, 'y')
    st.plotly_chart(accel_gauge_y)

    st.subheader("Aceleração Z")
    accel_gauge_z = create_acceleration_gauge(data, 'z')
    st.plotly_chart(accel_gauge_z)

    st.subheader("Velocidade X")
    velocity_gauge_x = create_velocity_gauge(data, 'x')
    st.plotly_chart(velocity_gauge_x)

    st.subheader("Velocidade Y")
    velocity_gauge_y = create_velocity_gauge(data, 'y')
    st.plotly_chart(velocity_gauge_y)

    st.subheader("Velocidade Z")
    velocity_gauge_z = create_velocity_gauge(data, 'z')
    st.plotly_chart(velocity_gauge_z)
    
    # Adicionar uma coluna fictícia de potência (exemplo: 50 CV)
    data['power'] = 50
    
    # Preparar os dados para previsão (normalização)
    data_normalized = (data - data.mean(axis=0)) / data.std(axis=0)
    test_data = data_normalized.values[:, :-2]  # Convertendo para numpy array
    
    # Converter os dados para o tipo aceito pelo modelo (float32)
    test_data = test_data.astype(np.float32)
    
    # Fazer previsões
    predictions = model.predict(test_data)
    
    # Classificar as previsões conforme a norma
    power_column = 'power'  # Nome da coluna que contém a informação de potência
    data['predictions'] = predictions
    data['classification'] = data.apply(lambda row: classify_vibration(row['predictions'], row[power_column]), axis=1)
    
    st.write("Previsões e Classificações:")
    st.write(data[['timestamp', 'predictions', 'classification']])
    
    predictions_bar_chart = create_predictions_bar_chart(predictions)
    st.plotly_chart(predictions_bar_chart)
    
    classification_line_chart = create_classification_line_chart(data)
    st.plotly_chart(classification_line_chart)

# Adiciona a funcionalidade de atualização automática
st_autorefresh_interval = st.sidebar.slider("Intervalo de Atualização (segundos)", 10, 300, 60)
st_autorefresh = st.sidebar.checkbox("Ativar Atualização Automática", value=True)

if st_autorefresh:
    import time
    time.sleep(st_autorefresh_interval)
    st.experimental_rerun()