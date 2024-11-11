import os
import pytorch_lightning as pl
from omegaconf import OmegaConf
import nemo.collections.asr as nemo_asr
from nemo.utils.exp_manager import exp_manager
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torch

# Assicurati di utilizzare alta precisione per moltiplicazioni
torch.set_float32_matmul_precision('high')

# Carica il file di configurazione
cfg = OmegaConf.load("/home/giolinux/NLP_Project/config.yaml")

# Imposta i checkpoint per salvare i migliori modelli
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    mode="min",
    save_top_k=3,
    dirpath="./experiments/checkpoints",
    filename="asr-{epoch:02d}-{val_loss:.2f}"
)

# Trainer con configurazione base per semplificare il modello
trainer = pl.Trainer(
    max_epochs=cfg.trainer.max_epochs,
    accelerator=cfg.trainer.accelerator,
    devices=cfg.trainer.devices,
    callbacks=[
        EarlyStopping(monitor="val_loss", patience=7, mode="min"),
        checkpoint_callback
    ],
    logger=False  # Disabilita il logger predefinito
)

# Configura l'exp_manager
exp_manager(trainer, cfg.get("exp_manager", None))

# Inizializza il modello con la configurazione di base
asr_model = nemo_asr.models.EncDecClassificationModel(cfg=cfg.model)

# Inizializza manualmente l'ottimizzatore per evitarne la mancanza
asr_model._optimizer = torch.optim.Adam(asr_model.parameters(), lr=1e-4)

# Imposta i dati di addestramento e validazione
asr_model.setup_training_data(train_data_config=cfg.model.train_ds)
asr_model.setup_validation_data(val_data_config=cfg.model.validation_ds)

# Avvia l'addestramento
trainer.fit(asr_model)

# Valuta il modello dopo l'addestramento
trainer.validate(asr_model)

# Salva il modello addestrato
model_save_path = "/home/giolinux/NLP_Project/asr_model_simple.nemo"
asr_model.save_to(model_save_path)

print(f"Training completato, modello salvato in {model_save_path}")