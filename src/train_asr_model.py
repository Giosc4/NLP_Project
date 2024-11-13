import os
import json
import librosa
import random
import pytorch_lightning as pl
from omegaconf import OmegaConf
from nemo.collections.asr.models import EncDecClassificationModel
from nemo.utils.exp_manager import exp_manager
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torch

# Assicurati di utilizzare alta precisione per moltiplicazioni
torch.set_float32_matmul_precision('high')

# Funzione per dividere il manifest in train e validation
def split_manifest(manifest_path, train_manifest_path, val_manifest_path, train_split=0.8):
    # Carica il manifest
    with open(manifest_path, 'r') as f:
        data = f.readlines()

    # Miscelare casualmente i dati
    random.shuffle(data)

    # Calcola la divisione per l'80% training e 20% validation
    split_idx = int(len(data) * train_split)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    # Salva i dati di training
    with open(train_manifest_path, 'w') as f_train:
        f_train.writelines(train_data)

    # Salva i dati di validazione
    with open(val_manifest_path, 'w') as f_val:
        f_val.writelines(val_data)

    print(f"Divisione completata: {len(train_data)} campioni per il training, {len(val_data)} campioni per la validazione.")

# Funzione per estrarre l'etichetta dal nome del file
def extract_label_from_filename(filepath):
    filename = os.path.basename(filepath)
    parts = filename.split('_')
    if len(parts) >= 2:
        label = parts[1]
    else:
        # Se il nome del file non segue il pattern previsto, prendi il nome del file senza estensione
        label = os.path.splitext(filename)[0]
    # Rimuovi eventuali estensioni o suffissi
    label = label.replace('.wav', '').replace('.mp3', '')
    return label

# Funzione per creare il manifest solo per i file non aumentati
def create_manifest_non_augmented(data_dir, manifest_path, augmented_dir='augmented'):
    with open(manifest_path, 'w') as manifest_file:
        for root, _, files in os.walk(data_dir):
            if augmented_dir in root:
                continue  # Escludi i file aumentati
            for file in files:
                if file.endswith('.wav'):
                    file_path = os.path.join(root, file)
                    label = extract_label_from_filename(file_path)
                    duration = librosa.get_duration(path=file_path)
                    entry = {
                        'audio_filepath': file_path,
                        'duration': duration,
                        'label': label
                    }
                    manifest_file.write(json.dumps(entry) + '\n')

# Carica il file di configurazione aggiornato
cfg = OmegaConf.load("/home/giolinux/NLP_Project/config.yaml")

# Crea il manifest file solo per file non aumentati
data_dir = cfg.model.train_ds.get("data_dir", "/home/giolinux/NLP_Project/audio")  # Directory dei file non aumentati
manifest_path = "/home/giolinux/NLP_Project/data_manifest.json"
train_manifest_path = cfg.model.train_ds.get("manifest_filepath", "/home/giolinux/NLP_Project/train_manifest.json")
val_manifest_path = cfg.model.validation_ds.get("manifest_filepath", "/home/giolinux/NLP_Project/val_manifest.json")

create_manifest_non_augmented(data_dir, manifest_path)

# Dividi il manifest (comprendendo anche i dati augmented) in train e val
split_manifest(manifest_path, train_manifest_path, val_manifest_path)

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
exp_manager(trainer, cfg.exp_manager)

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
model_save_path = "/home/giolinux/NLP_Project/asr_model.nemo"
asr_model.save_to(model_save_path)

print(f"Training completato, modello salvato in {model_save_path}")