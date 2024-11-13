import os
import optuna
import pytorch_lightning as pl
from omegaconf import OmegaConf
from nemo.collections.asr.models import EncDecClassificationModel
from nemo.utils.exp_manager import exp_manager
import torch

def objective(trial):
    # Carica la configurazione
    cfg = OmegaConf.load("/home/giolinux/NLP_Project/config.yaml")

    # Scegli gli iperparametri da ottimizzare
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    dropout = trial.suggest_uniform('dropout', 0.1, 0.5)
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16, 32])
    filters = trial.suggest_categorical('filters', [64, 128, 256])

    # Aggiorna la configurazione con gli iperparametri scelti
    cfg.model.optim.lr = lr
    cfg.model.encoder.jasper[0].dropout = dropout
    cfg.model.encoder.jasper[1].dropout = dropout
    cfg.model.encoder.jasper[0].filters = filters
    cfg.model.encoder.jasper[1].filters = filters
    cfg.model.decoder.feat_in = filters
    cfg.model.train_ds.batch_size = batch_size
    cfg.model.validation_ds.batch_size = batch_size

    # Disabilita il checkpointing e il logging per velocizzare l'ottimizzazione
    cfg.trainer.enable_checkpointing = False
    cfg.trainer.logger = False
    cfg.exp_manager.create_checkpoint_callback = False
    cfg.exp_manager.create_tensorboard_logger = False

    # Imposta il numero di epoch per una valutazione rapida
    cfg.trainer.max_epochs = 100

    # Imposta il trainer senza logger e checkpointing
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        logger=False,
        enable_checkpointing=False
    )

    # Configura l'exp_manager
    exp_manager(trainer, cfg.exp_manager)

    # Inizializza il modello
    asr_model = EncDecClassificationModel(cfg=cfg.model)

    # Imposta i dati
    asr_model.setup_training_data(train_data_config=cfg.model.train_ds)
    asr_model.setup_validation_data(val_data_config=cfg.model.validation_ds)

    # Avvia l'addestramento
    trainer.fit(asr_model)

    # Valuta il modello
    result = trainer.validate(asr_model)

    # Ottieni la metrica da ottimizzare (ad esempio, val_loss)
    val_loss = result[0]['val_loss']

    # Restituisci la metrica per l'ottimizzazione
    return val_loss

if __name__ == '__main__':
    # Imposta la precisione delle moltiplicazioni
    torch.set_float32_matmul_precision('high')

    # Specifica il percorso per il database SQLite
    storage_name = "sqlite:///asr_study.db"

    # Crea uno studio Optuna con storage persistente
    study = optuna.create_study(
        direction='minimize',
        study_name='ASR_Hyperparameter_Optimization',
        storage=storage_name,
        load_if_exists=True
    )

    # Esegui l'ottimizzazione
    study.optimize(objective, n_trials=5)

    # Stampa i migliori iperparametri trovati
    print("I migliori iperparametri trovati sono:")
    print(study.best_params)
