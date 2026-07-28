import time
from prometheus_client import start_http_server, Counter

FRAUD_COUNTER = Counter('fraud_calls_total', 'Gesamtanzahl erkannter Betrugsversuche')

def start_exporter(port: int = 8000):
    start_http_server(port)
    print(f"Prometheus Exporter laeuft auf Port {port}")
    while True:
        time.sleep(1)
