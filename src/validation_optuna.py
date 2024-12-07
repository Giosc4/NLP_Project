import os
import json
import torch
import optuna
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from omegaconf import OmegaConf
from nemo.collections.asr.models import EncDecClassificationModel
from nemo.collections.asr.metrics.wer import word_error_rate
from difflib import SequenceMatcher

###############################################
# Carica la configurazione e modifica parametri
###############################################
cfg = OmegaConf.load("../config.yaml")

# Aumenta il numero di epoche per permettere un training più lungo
cfg.trainer.max_epochs = 50

# Se vuoi cambiare i manifest direttamente da codice (opzionale):
# cfg.model.train_ds.manifest_filepath = "../train_manifest.json"
# cfg.model.validation_ds.manifest_filepath = "../val_manifest.json"
# cfg.model.test_ds.manifest_filepath = "../test_manifest.json" 
# Assicurati di aver definito test_ds nel config se vuoi usare un test set separato.

###############################################
# Funzione Objective per Optuna
###############################################
def objective(trial):
    # Ricerca iperparametri con Optuna
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

    cfg.model.optim.lr = lr
    cfg.model.train_ds.batch_size = batch_size

    # Istanzia il modello
    asr_model = EncDecClassificationModel(cfg=cfg.model)

    # Setup dei dati di training e validazione
    asr_model.setup_training_data(train_data_config=cfg.model.train_ds)
    asr_model.setup_validation_data(val_data_config=cfg.model.validation_ds)

    # Early Stopping per evitare overfitting
    early_stopping = EarlyStopping(monitor="val_epoch_top@1", patience=5, mode="max")

    # Non salviamo qui i checkpoint, lo faremo nel final training, ma si potrebbe già fare
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        callbacks=[early_stopping],
        logger=False
    )

    trainer.fit(asr_model)
    metrics = trainer.validate(asr_model)
    val_score = metrics[0]['val_epoch_top@1']
    return val_score

###############################################
# Esecuzione di Optuna per l'hyperparameter tuning
###############################################
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)  # Aumentiamo i trial rispetto ai 10 precedenti

print(f"Best hyperparameters: {study.best_params}")
print(f"Best value: {study.best_value}")

# Aggiorna config con i migliori iperparametri trovati
cfg.model.optim.lr = study.best_params["learning_rate"]
cfg.model.train_ds.batch_size = study.best_params["batch_size"]

###############################################
# Training Finale con i Migliori Iperparametri
# + EarlyStopping e ModelCheckpoint
###############################################
final_model = EncDecClassificationModel(cfg=cfg.model)
final_model.setup_training_data(train_data_config=cfg.model.train_ds)
final_model.setup_validation_data(val_data_config=cfg.model.validation_ds)

# Callback per salvare i migliori modelli
checkpoint_callback = ModelCheckpoint(
    monitor="val_epoch_top@1",
    mode="max",
    save_top_k=3,
    filename='asr-{epoch:02d}-{val_epoch_top@1:.2f}'
)

early_stopping = EarlyStopping(monitor="val_epoch_top@1", patience=5, mode="max")

final_trainer = pl.Trainer(
    max_epochs=cfg.trainer.max_epochs,
    accelerator=cfg.trainer.accelerator,
    devices=cfg.trainer.devices,
    callbacks=[early_stopping, checkpoint_callback],
    logger=False
)

final_trainer.fit(final_model)
final_metrics = final_trainer.validate(final_model)
print("Final model validation performance:", final_metrics)

# Se hai un test set separato, valuta anche sul test set
# Assicurati che cfg.model.test_ds sia definito nel config.yaml
if hasattr(cfg.model, 'test_ds'):
    final_model.setup_test_data(test_data_config=cfg.model.test_ds)
    test_metrics = final_trainer.test(final_model)
    print("Test set performance:", test_metrics)

# Salva il modello finale in formato .nemo
model_save_path = "../asr_model.nemo"
final_model.save_to(model_save_path)
print(f"Final trained model saved at {model_save_path}")

###############################################
# Classe di Inference per Valutazione Finale
###############################################
import nemo.collections.asr as nemo_asr

class ASRInference:
    def __init__(self, model_path, val_manifest, use_gpu=True):
        self.model = nemo_asr.models.EncDecClassificationModel.restore_from(model_path)
        self.audio_paths = self.load_audio_paths(val_manifest)
        self.labels = self.model.cfg.labels
        self.model.eval()
        if use_gpu and torch.cuda.is_available():
            self.model.cuda()

    @staticmethod
    def load_audio_paths(manifest_path):
        with open(manifest_path, 'r') as f:
            return [json.loads(line)["audio_filepath"] for line in f]

    def transcribe_audio(self):
        transcriptions = []
        with torch.no_grad():
            for audio_path in self.audio_paths:
                transcription = self.model.transcribe([audio_path])
                if isinstance(transcription[0], torch.Tensor):
                    index = transcription[0].item()
                    predicted_label = self.labels[index] if 0 <= index < len(self.labels) else "indice_non_valido"
                else:
                    predicted_label = transcription[0]
                print(f"Raw model output for {audio_path}: {transcription}, Predicted Label: {predicted_label}")
                transcriptions.append(predicted_label)
        return transcriptions

    def extract_label_from_path(self, path):
        base = os.path.basename(path)
        try:
            label = base.split('_')[1].split('.')[0]
            return label.lower()
        except IndexError:
            print(f"Error extracting label from file: {base}")
            return None

    def calculate_cer(self, hypotheses, references):
        total_chars = sum(len(ref) for ref in references)
        total_errors = sum(
            len(ref) + len(hyp) - 2 * sum(block.size for block in SequenceMatcher(None, ref, hyp).get_matching_blocks())
            for ref, hyp in zip(references, hypotheses)
        )
        return total_errors / total_chars if total_chars > 0 else 0

    def display_results(self):
        print("========================")
        print("Inizio Trascrizione e Valutazione")
        print("========================")
        
        transcriptions = self.transcribe_audio()
        correct = []
        incorrect = []

        for idx, (path, transcription) in enumerate(zip(self.audio_paths, transcriptions), start=1):
            ground_truth = self.extract_label_from_path(path)
            if ground_truth is None:
                continue

            predicted_label = transcription.lower() if isinstance(transcription, str) else "formato_non_riconosciuto"

            if predicted_label == ground_truth:
                correct.append((path, predicted_label))
            else:
                incorrect.append((path, ground_truth, predicted_label))

            print(f"File {idx}: {path}")
            print(f"   Ground Truth: {ground_truth}")
            print(f"   Prediction:   {predicted_label}")
            print("---------------------------")

        total = len(correct) + len(incorrect)
        accuracy = (len(correct) / total) * 100 if total > 0 else 0

        print("========================")
        print("Risultati Generali")
        print("========================")
        print(f"Total files analyzed: {total}")
        print(f"Correct predictions: {len(correct)}")
        print(f"Incorrect predictions: {len(incorrect)}")
        print(f"Accuracy percentage: {accuracy:.2f}%\n")

        ground_truths = [self.extract_label_from_path(path) for path in self.audio_paths]
        wer = word_error_rate(hypotheses=transcriptions, references=ground_truths)
        cer = self.calculate_cer(transcriptions, ground_truths)

        print("========================")
        print("Metriche di Valutazione")
        print("========================")
        print(f"Word Error Rate (WER): {wer:.2f}")
        print(f"Character Error Rate (CER): {cer:.2f}")

###############################################
# Esempio di Inference con il modello finale
###############################################
inference = ASRInference(model_path, "../val_manifest.json")
inference.display_results()
