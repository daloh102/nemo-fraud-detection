import requests
import json

print("🔍 Starte System-Validation & Smoke Test...\n")

# 1. Prüfe LLM Inference Server (NIM)
try:
    nim_res = requests.get("http://localhost:8000/v1/health/ready", timeout=5)
    if nim_res.status_code == 200:
        print("✅ 1. NIM LLM Inference Server: BEREIT (Port 8000)")
    else:
        print(f"⚠️ 1. NIM LLM antwortet, aber nicht ready Status Code: {nim_res.status_code}")
except Exception as e:
    print(f"❌ 1. NIM LLM nicht erreichbar! Lädt das Modell noch in den VRAM? ({e})")

# 2. Prüfe NeMo Data Designer Microservice
try:
    # Einfacher Ping an den Data Designer
    dd_res = requests.get("http://localhost:8004/v1/healthcheck", timeout=5) # bzw. Wurzel-Endpunkt
    print("✅ 2. NeMo Data Designer Microservice: BEREIT (Port 8004)")
except Exception:
    # Alternativ: Test über OpenAPI Docs/Schema
    try:
        dd_res = requests.get("http://localhost:8004/docs", timeout=5)
        print("✅ 2. NeMo Data Designer Microservice: BEREIT (Port 8004 / Docs erreichbar)")
    except Exception as e:
        print(f"❌ 2. NeMo Data Designer nicht erreichbar! ({e})")

# 3. Teste Inferenz-Generierung mit einem Test-Prompt via NIM
try:
    payload = {
        "model": "mistralai/mistral-medium-3.5-128b",
        "messages": [{"role": "user", "content": "Hallo! Antworte kurz: Bist du online?"}],
        "max_tokens": 20
    }
    nim_gen = requests.post("http://localhost:8000/v1/chat/completions", json=payload, timeout=30)
    if nim_gen.status_code == 200:
        answer = nim_gen.json()['choices'][0]['message']['content']
        print(f"✅ 3. Inferenz-Test erfolgreich! Modell-Antwort: '{answer.strip()}'")
    else:
        print(f"❌ 3. Inferenz-Test fehlgeschlagen mit Status {nim_gen.status_code}")
except Exception as e:
    print(f"❌ 3. Inferenz-Test abgebrochen: {e}")