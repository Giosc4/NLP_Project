import os
import optuna
import pytorch_lightning as pl
from omegaconf import OmegaConf
from nemo.collections.asr.models import EncDecClassificationModel
from pytorch_lightning.callbacks import EarlyStopping
import torch

def log_metric(epoch, metric, value):
    print(f"Epoch {epoch}, {metric}: {value}")

def objective(trial):
    # Carica la configurazione
    cfg = OmegaConf.load("../config.yaml")

    # Imposta il seed per la riproducibilità
    pl.seed_everything(42)

    # Definisci iperparametri da ottimizzare
    learning_rate = trial.suggest_loguniform("lr", 1e-5, 1e-3)
    dropout = trial.suggest_uniform("dropout", 0.1, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    num_layers = trial.suggest_int("num_layers", 2, 5)
    filters = trial.suggest_categorical("filters", [64, 128, 256])
    kernel_size = trial.suggest_int("kernel_size", 3, 11, step=2)

    # Aggiorna la configurazione solo con i parametri ottimizzati
    cfg.model.optim.lr = learning_rate
    cfg.model.encoder.jasper = []
    for i in range(num_layers):
        cfg.model.encoder.jasper.append({
            'filters': filters,
            'repeat': 1,
            'kernel': [kernel_size],
            'stride': [1],
            'dilation': [1],
            'dropout': dropout,
            'residual': True if i > 0 else False,
            'activation': 'relu',
        })

    cfg.model.train_ds.batch_size = batch_size
    cfg.model.validation_ds.batch_size = batch_size

    # Disabilita il checkpointing e il logging per velocizzare l'ottimizzazione
    cfg.trainer.enable_checkpointing = False
    cfg.trainer.logger = False

    # Imposta il numero di epoch per una valutazione rapida
    cfg.trainer.max_epochs = 10

    # Callback per Early Stopping
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=False,
        mode='min'
    )

    # Configura il trainer
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        callbacks=[early_stop_callback],
        precision=16  # Abilita precisione mista
    )

    # Inizializza il modello
    asr_model = EncDecClassificationModel(cfg=cfg.model)

    # Imposta i dati
    asr_model.setup_training_data(train_data_config=cfg.model.train_ds)
    asr_model.setup_validation_data(val_data_config=cfg.model.validation_ds)

    # Training e valutazione
    try:
        trainer.fit(asr_model)
    except Exception as e:
        print(f"Errore durante l'addestramento: {e}")
        return float('inf')

    # Valuta il modello
    result = trainer.validate(asr_model)

    # Ottieni la metrica da ottimizzare (ad esempio, val_loss)
    val_loss = result[0]['val_loss']

    # Logging del risultato
    print(f"Trial {trial.number} - Val Loss: {val_loss}")

    # Pruning
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return val_loss


if __name__ == '__main__':
    # Imposta precisione matematica elevata
    torch.set_float32_matmul_precision('high')

    # Specifica il percorso per il database SQLite
    storage_name = "sqlite:///asr_study.db"

    # Crea uno studio Optuna
    study = optuna.create_study(
        direction='minimize',
        study_name='ASR_Hyperparameter_Optimization',
        storage=storage_name,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner()
    )

    # Esegui l'ottimizzazione
    study.optimize(objective, n_trials=10)

    # Stampa i migliori iperparametri trovati
    print("I migliori iperparametri trovati sono:")
    print(study.best_params)
