from flask import Flask, request, jsonify
import paho.mqtt.client as mqtt
import psycopg2
import json

app = Flask(__name__)

# Configurações do MQTT
MQTT_BROKER = 'test.mosquitto.org'
MQTT_PORT = 1883
MQTT_TOPIC = 'iot_monit_motor'

# Configurações do banco de dados PostgreSQL
DB_HOST = 'localhost'
DB_NAME = 'iotmotorhealth'
DB_USER = 'postgres'
DB_PASS = '123456'

# Função para conectar ao PostgreSQL
def connect_db():
    conn = psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST
    )
    return conn

# Callback que será chamado quando uma mensagem for recebida do broker MQTT
def on_message(client, userdata, message):
    payload = message.payload.decode('utf-8')
    data = json.loads(payload)

    conn = connect_db()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO iotmotorhealth (aceleracao_x, aceleracao_y, aceleracao_z, velocidade_x, velocidade_y, velocidade_z, temperatura)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (
        data['aceleracao']['x'], data['aceleracao']['y'], data['aceleracao']['z'],
        data['velocidade']['x'], data['velocidade']['y'], data['velocidade']['z'],
        data['temperatura']
    ))
    conn.commit()
    cur.close()
    conn.close()

# Configurar o cliente MQTT
mqtt_client = mqtt.Client()
mqtt_client.on_message = on_message
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
mqtt_client.subscribe(MQTT_TOPIC)
mqtt_client.loop_start()

@app.route('/')
def index():
    return "MQTT to PostgreSQL bridge is running!"

if __name__ == '__main__':
    app.run(debug=True, port=5000)