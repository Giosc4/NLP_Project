import os
import json
import pytorch_lightning as pl
from omegaconf import OmegaConf
from nemo.collections.asr.models import EncDecClassificationModel
#from nemo.utils.exp_manager import exp_manager
from nemo.collections.asr.metrics.wer import word_error_rate
from pytorch_lightning.callbacks import EarlyStopping
from difflib import SequenceMatcher

import torch
import subprocess

# Esegui il primo script di data augmentation
script1 = 'data_augmentation.py'
result1 = subprocess.run(['python3', script1], capture_output=True, text=True)
print(f"Uscita di {script1}:\n{result1.stdout}")
if result1.stderr:
    print(f"Errori di {script1}:\n{result1.stderr}")

# Imposta alta precisione per le moltiplicazioni
torch.set_float32_matmul_precision('high')

# Carica il file di configurazione aggiornato
cfg = OmegaConf.load("../config.yaml")

# Imposta il manifest per il training e la validazione
cfg.model.train_ds.manifest_filepath = "../train_manifest_augmented.json"
cfg.model.validation_ds.manifest_filepath = "../val_manifest.json"


def calculate_cer(hypotheses, references):
    """
    Calculate Character Error Rate (CER) given hypotheses and references.
    CER = (S + D + I) / N
    where:
        S = number of substitutions
        D = number of deletions
        I = number of insertions
        N = total number of characters in the reference
    """
    total_chars = 0
    total_errors = 0

    for hyp, ref in zip(hypotheses, references):
        total_chars += len(ref)
        matcher = SequenceMatcher(None, ref, hyp)
        total_errors += sum([tag[1] for tag in matcher.get_opcodes() if tag[0] != 'equal'])

    cer = total_errors / total_chars if total_chars > 0 else 0
    return cer

class WERandCERCallback(pl.Callback):
    def __init__(self, val_manifest):
        super().__init__()
        self.val_manifest = val_manifest

    def on_validation_end(self, trainer, pl_module):
        # Carica il manifest di validazione
        with open(self.val_manifest, 'r') as f:
            val_data = [json.loads(line) for line in f]

        predictions = []
        ground_truths = []

        # Genera trascrizioni per ciascun file
        for item in val_data:
            audio_path = item["audio_filepath"]
            ground_truth = item["label"]

            # Ottieni la predizione dal modello
            predicted_text = pl_module.transcribe([audio_path])[0]

            # Debug: verifica il tipo di predizione
            print(f"Raw Predicted Text: {predicted_text} (Type: {type(predicted_text)})")

            # Converti in stringa, se necessario
            if isinstance(predicted_text, int):  # Caso indice numerico
                predicted_text = pl_module.labels[predicted_text]
            elif isinstance(predicted_text, list):  # Caso lista di indici
                predicted_text = " ".join(map(str, predicted_text))
            elif isinstance(predicted_text, torch.Tensor):  # Caso tensore
                predicted_text = " ".join(map(str, predicted_text.cpu().numpy()))

            predictions.append(predicted_text)
            ground_truths.append(str(ground_truth))

        # Debug: verifica il contenuto delle liste
        print("Predictions:", predictions)
        print("Ground Truths:", ground_truths)

        # Verifica che tutti gli elementi siano stringhe
        assert all(isinstance(p, str) for p in predictions), "All predictions must be strings"
        assert all(isinstance(g, str) for g in ground_truths), "All ground truths must be strings"

        # Calcola WER e CER
        wer = word_error_rate(hypotheses=predictions, references=ground_truths)
        cer = calculate_cer(predictions, ground_truths)

        print(f"Validation WER: {wer:.2f}")
        print(f"Validation CER: {cer:.2f}")


trainer = pl.Trainer(
    max_epochs=cfg.trainer.max_epochs,
    accelerator=cfg.trainer.accelerator,
    devices=cfg.trainer.devices,
    callbacks=[
        EarlyStopping(monitor="val_epoch_top@1", patience=5, mode="min"),
        WERandCERCallback(val_manifest=cfg.model.validation_ds.manifest_filepath),
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
model_save_path = "../asr_model.nemo"
asr_model.save_to(model_save_path)

print(f"Training completato, modello salvato in {model_save_path}")

# Esegui il secondo script
script2 = 'evaluate_model.py'
result2 = subprocess.run(['python3', script2], capture_output=True, text=True)
print(f"Uscita di {script2}:\n{result2.stdout}")
if result2.stderr:
    print(f"Errori di {script2}:\n{result2.stderr}")
