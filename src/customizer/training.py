import pytorch_lightning as pl
from omegaconf import DictConfig
from nemo.collections.nlp.models.language_modeling.megatron_gpt_sft_model import MegatronGPTSFTModel

def run_lora_training(cfg: DictConfig):
    print(f"Lade Basismodell von: {cfg.model.restore_from_path}")
    print(f"Initialisiere PEFT-Schema: {cfg.model.peft.peft_scheme}")
    return True
