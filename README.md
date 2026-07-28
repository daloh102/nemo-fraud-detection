\# 🛡️ Bank Fraud Detection with NVIDIA NeMo



Dieses Repository enthält eine vollständige MLOps-Pipeline zur Erkennung von Betrugsversuchen in Telefon-Transkripten. Das Projekt nutzt das \*\*NVIDIA NeMo-Framework\*\* für das Kuratieren von Datensätzen, das effiziente Fine-Tuning mittels PEFT (LoRA) sowie die automatisierte Evaluation und produktive Überwachung.



\---



\## 🏗️ Projekt-Architektur \& Workflow



Die Pipeline folgt einem strengen 5-Stufen-Modell, das sich in der Ordnerstruktur widerspiegelt:



```text

bank\_fraud\_nemo\_project/

├── configs/            # Zentralisierte YAML-Konfigurationen für jede Phase

├── data/               # Daten-Pipeline (Raw -> Interim -> Curated -> SFT)

├── monitoring/         # Prometheus \& Grafana Infrastruktur-Stubs

├── notebooks/          # Interaktive Experimente \& Pipeline-Schritte

├── src/                # Modularer, produktionsreifer Python-Code

└── reports/            # Generierte Audit- und Qualitätsberichte

