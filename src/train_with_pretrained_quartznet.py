import optuna
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from nemo.collections.asr.models import EncDecClassificationModel
from omegaconf import OmegaConf

# Carica la configurazione
cfg = OmegaConf.load("../config.yaml")  # Sostituisci con il percorso corretto al tuo config.yaml

def objective(trial):
    # Scegli gli iperparametri da ottimizzare
    # Usa suggest_float con log=True invece di suggest_loguniform
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

    # Aggiorna la config con gli iperparametri proposti
    cfg.model.optim.lr = lr
    cfg.model.train_ds.batch_size = batch_size

    # Reinizializza il modello con questi parametri
    asr_model = EncDecClassificationModel(cfg=cfg.model)

    # Aggiorna i dati di training e validazione
    asr_model.setup_training_data(train_data_config=cfg.model.train_ds)
    asr_model.setup_validation_data(val_data_config=cfg.model.validation_ds)

    # Callbacks: early stopping per evitare di sprecare risorse in trial non promettenti
    early_stopping = EarlyStopping(monitor="val_epoch_top@1", patience=5, mode="max")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        callbacks=[early_stopping],
        logger=False
    )

    # Addestra
    trainer.fit(asr_model)

    # Valuta sul validation set
    metrics = trainer.validate(asr_model)

    # Supponendo che 'val_epoch_top@1' sia la metrica di interesse
    val_score = metrics[0]['val_epoch_top@1']

    # Ritorna il punteggio da massimizzare
    return val_score

# Crea uno study Optuna
study = optuna.create_study(direction="maximize")  
study.optimize(objective, n_trials=10)  # Esegui 10 trial

print(f"Best hyperparameters: {study.best_params}")
print(f"Best value: {study.best_value}")

# Dopo aver trovato i migliori iperparametri, aggiorna la config
cfg.model.optim.lr = study.best_params["learning_rate"]
cfg.model.train_ds.batch_size = study.best_params["batch_size"]

final_model = EncDecClassificationModel(cfg=cfg.model)
final_model.setup_training_data(train_data_config=cfg.model.train_ds)
final_model.setup_validation_data(val_data_config=cfg.model.validation_ds)

final_trainer = pl.Trainer(
    max_epochs=cfg.trainer.max_epochs,
    accelerator=cfg.trainer.accelerator,
    devices=cfg.trainer.devices,
    callbacks=[EarlyStopping(monitor="val_epoch_top@1", patience=5, mode="max")],
    logger=False
)

final_trainer.fit(final_model)
final_metrics = final_trainer.validate(final_model)
print("Final model performance:", final_metrics)
