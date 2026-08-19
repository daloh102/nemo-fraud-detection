FROM nvcr.io/nvidia/nemo:24.09

# Installation von zusätzlichen Paketen für Guardrails, Tracking, Evaluierung und Training
RUN pip install --no-cache-dir \
    nemoguardrails \
    wandb \
    protobuf \
    requests \
    mlflow \
    scikit-learn 
    
# Upgrade der übrigen Pakete
RUN pip install --upgrade wandb protobuf mlflow scikit-learn

# Arbeitsverzeichnis setzen
WORKDIR /data