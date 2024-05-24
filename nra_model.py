import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import psycopg2
import tensorflow as tf
from keras.models import save_model
from scipy.fftpack import fft
from scipy.signal import find_peaks
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
def get_data():
    conn = get_connection()
    query = "SELECT * FROM iotmotorhealth"
    data = pd.read_sql(query, conn)
    conn.close()
    return data

# Função para classificar o nível de vibração conforme a norma ISO 10816 ou ISO 2372
def classify_vibration(value, power):
    if power <= 20:
        if value <= 0.28:
            return "BOM"
        elif value <= 0.45:
            return "ADEQUADO"
        elif value <= 0.71:
            return "ADEQUADO"
        elif value <= 1.12:
            return "ADMISSÍVEL"
        elif value <= 1.8:
            return "ADMISSÍVEL"
        else:
            return "INADMISSÍVEL"
    elif power <= 100:
        if value <= 0.45:
            return "BOM"
        elif value <= 0.71:
            return "ADEQUADO"
        elif value <= 1.12:
            return "ADMISSÍVEL"
        elif value <= 1.8:
            return "ADMISSÍVEL"
        elif value <= 2.8:
            return "ADMISSÍVEL"
        else:
            return "INADMISSÍVEL"
    else:
        if value <= 0.71:
            return "BOM"
        elif value <= 1.12:
            return "ADEQUADO"
        elif value <= 1.8:
            return "ADEQUADO"
        elif value <= 2.8:
            return "ADMISSÍVEL"
        elif value <= 4.5:
            return "ADMISSÍVEL"
        else:
            return "INADMISSÍVEL"

# Função para calcular a FFT
def apply_fft(data, sampling_rate):
    data = data.to_numpy()  # Converte a série Pandas para um array NumPy
    n = len(data)
    T = 1.0 / sampling_rate
    yf = fft(data)
    xf = np.linspace(0.0, 1.0/(2.0*T), n//2)
    return xf, 2.0/n * np.abs(yf[:n//2])

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

# Obter dados do banco de dados
data = get_data()

# Adicionar uma coluna fictícia de potência (exemplo: 50 CV)
data['power'] = 50

# Verificar colunas disponíveis no DataFrame
print("Colunas disponíveis no DataFrame:", data.columns)

# Nome da coluna que contém a informação de potência
power_column = 'power'

# Normaliza os dados
data_normalized = (data - data.mean(axis=0)) / data.std(axis=0)

# Separa os dados em conjuntos de treinamento e teste
train_data = data_normalized[:int(len(data)*0.8)]
test_data = data_normalized[int(len(data)*0.8):]

# Convertendo os DataFrames do pandas em arrays do NumPy e garantindo o tipo float32
X_train = train_data.iloc[:, :-2].values.astype(np.float32)
y_train = train_data.iloc[:, -2].values.astype(np.float32)
X_test = test_data.iloc[:, :-2].values.astype(np.float32)
y_test = test_data.iloc[:, -2].values.astype(np.float32)

# Define o modelo da rede neural
model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compila o modelo
model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001))

# Treina o modelo
history = model.fit(X_train, y_train, epochs=100, verbose=1)

# Salva o modelo treinado
save_model(model, 'motor_health_model.keras')

# Faz previsões no conjunto de teste
predictions = model.predict(X_test).astype(np.float32)

# Adicionar uma coluna de previsões ao DataFrame de teste
test_data['predictions'] = predictions

# Classificar as previsões conforme a norma
test_data['classification'] = test_data.apply(lambda row: classify_vibration(row['predictions'], row[power_column]), axis=1)

# Exibe as classificações
print(test_data[['predictions', 'classification']])

# Aplicar FFT e plotar espectros de frequência em uma única aba
sampling_rate = 800  # Exemplo de taxa de amostragem em Hz
all_figures = []

# Criar uma figura com subplots
fig = make_subplots(rows=2, cols=3, subplot_titles=("Aceleração X", "Aceleração Y", "Aceleração Z", "Velocidade X", "Velocidade Y", "Velocidade Z"))

# Função para adicionar espectro de frequência ao subplot
def add_spectrum_to_fig(fig, xf, yf, row, col, title):
    fig.add_trace(go.Scatter(x=xf, y=yf, mode='lines', name=title), row=row, col=col)

# Criar lista para armazenar classificações
classification_results = []

# Adicionar gráficos de aceleração
for i, axis in enumerate(['aceleracao_x', 'aceleracao_y', 'aceleracao_z']):
    xf, yf = apply_fft(data[axis], sampling_rate)
    peaks_x, peaks_y = identify_peaks(xf, yf)
    add_spectrum_to_fig(fig, xf, yf, 1, i + 1, f'{axis.upper()}')
    classifications = classify_spectrum(peaks_x, peaks_y)
    classification_results.append((axis, classifications))
    print(f"Classificação para {axis}: {classifications}")

# Adicionar gráficos de velocidade
for i, axis in enumerate(['velocidade_x', 'velocidade_y', 'velocidade_z']):
    xf, yf = apply_fft(data[axis], sampling_rate)
    peaks_x, peaks_y = identify_peaks(xf, yf)
    add_spectrum_to_fig(fig, xf, yf, 2, i + 1, f'{axis.upper()}')
    classifications = classify_spectrum(peaks_x, peaks_y)
    classification_results.append((axis, classifications))
    print(f"Classificação para {axis}: {classifications}")

# Atualizar layout da figura
fig.update_layout(height=800, width=1200, title_text="Espectros de Frequência")
fig.show()

# Criar tabela de classificações
classification_table = pd.DataFrame(classification_results, columns=['Axis', 'Classifications'])

# Exibir tabela de classificações
print("\nTabela de Classificações:")
print(classification_table)