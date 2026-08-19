FROM nvcr.io/nvidia/nemo:24.09

# Installation von zusätzlichen Paketen für Guardrails, Tracking, Evaluierung und Training
RUN pip install --no-cache-dir \
    nemoguardrails \
    wandb \
    protobuf \
    requests \
    mlflow \
    scikit-learn \
    bitsandbytes

# Erneuter, erzwungener Durchlauf für bitsandbytes, um saubere Paket-Metadaten sicherzustellen
RUN pip install --upgrade --force-reinstall bitsandbytes

# Upgrade der übrigen Pakete
RUN pip install --upgrade wandb protobuf mlflow scikit-learn
RUN pip install --upgrade --force-reinstall torchvision

# Arbeitsverzeichnis setzen
WORKDIR /data