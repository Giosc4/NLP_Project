import os
import json
import pytorch_lightning as pl
from omegaconf import OmegaConf
from nemo.collections.asr.models import EncDecClassificationModel
#from nemo.utils.exp_manager import exp_manager
from pytorch_lightning.callbacks import EarlyStopping
import torch
import subprocess

# Esegui il primo script di data augmentation
script1 = '/home/giova/NLP_Project/src/data_augmentation.py'
result1 = subprocess.run(['python3', script1], capture_output=True, text=True)
print(f"Uscita di {script1}:\n{result1.stdout}")
if result1.stderr:
    print(f"Errori di {script1}:\n{result1.stderr}")

# Imposta alta precisione per le moltiplicazioni
torch.set_float32_matmul_precision('high')

# Carica il file di configurazione aggiornato
cfg = OmegaConf.load("/home/giova/NLP_Project/config.yaml")

# Imposta il manifest per il training e la validazione
cfg.model.train_ds.manifest_filepath = "/home/giova/NLP_Project/train_manifest_augmented.json"
cfg.model.validation_ds.manifest_filepath = "/home/giova/NLP_Project/val_manifest.json"

# Configura il trainer
trainer = pl.Trainer(
    max_epochs=cfg.trainer.max_epochs,
    accelerator=cfg.trainer.accelerator,
    devices=cfg.trainer.devices,
    callbacks=[
        EarlyStopping(monitor="val_epoch_top@1", patience=7, mode="max")
    ],
    logger=False
)

# Configura l'exp_manager
#exp_manager(trainer, cfg.exp_manager)

# Inizializza il modello di classificazione
asr_model = EncDecClassificationModel(cfg=cfg.model)

# Imposta i dati di addestramento e validazione
asr_model.setup_training_data(train_data_config=cfg.model.train_ds)
asr_model.setup_validation_data(val_data_config=cfg.model.validation_ds)

# Avvia l'addestramento
trainer.fit(asr_model)

# Valuta il modello dopo l'addestramento
trainer.validate(asr_model)

# Salva il modello addestrato
model_save_path = "/home/giova/NLP_Project/asr_model.nemo"
asr_model.save_to(model_save_path)

print(f"Training completato, modello salvato in {model_save_path}")

# Esegui il secondo script
script2 = '/home/giova/NLP_Project/src/evaluate_model.py'
result2 = subprocess.run(['python3', script2], capture_output=True, text=True)
print(f"Uscita di {script2}:\n{result2.stdout}")
if result2.stderr:
    print(f"Errori di {script2}:\n{result2.stderr}")
