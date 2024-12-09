import time
start_time = time.time()

import optuna
import pytorch_lightning as pl
from nemo.collections.asr.models import EncDecClassificationModel
from omegaconf import OmegaConf
import subprocess

# Load the configuration file
cfg = OmegaConf.load("../config.yaml")

def objective(trial):
    # Suggerisci iperparametri
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    num_layers = trial.suggest_int('num_layers', 2, 5)
    filters = trial.suggest_categorical('filters', [64, 128, 256])
    kernel_size = trial.suggest_int('kernel_size', 3, 11, step=2)

    # Ripristina la configurazione iniziale dell'encoder se necessario (per evitare accumuli di layer tra trial)
    # È importante perché stai appendendo i layer ad ogni trial. Assicurati che l'encoder parta sempre dallo stato iniziale.
    # Ad esempio, se la config iniziale ha già due layer, rimuovi i layer aggiunti nei trial precedenti:
    base_jasper = [
        {
            'filters': 64,
            'repeat': 1,
            'kernel': [11],
            'stride': [1],
            'dilation': [1],
            'dropout': 0.3,
            'residual': False
        },
        {
            'filters': 128,
            'repeat': 1,
            'kernel': [13],
            'stride': [1],
            'dilation': [1],
            'dropout': 0.3,
            'residual': True
        }
    ]

    # Ripristina la configurazione iniziale
    cfg.model.encoder.jasper = base_jasper

    # Imposta gli iperparametri scelti da Optuna
    cfg.model.optim.lr = learning_rate
    cfg.model.train_ds.batch_size = batch_size

    # Aggiungi i nuovi layer all'encoder
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

    # Il numero di filtri finali dell'encoder sarà l'ultimo 'filters' aggiunto
    final_filters = filters

    # Aggiorna la dimensione del decoder
    cfg.model.decoder.feat_in = final_filters

    # Inizializza il modello con la config aggiornata
    asr_model = EncDecClassificationModel(cfg=cfg.model)

    # Set up training and validation data
    asr_model.setup_training_data(train_data_config=cfg.model.train_ds)
    asr_model.setup_validation_data(val_data_config=cfg.model.validation_ds)

    # Trainer setup
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        callbacks=[],
        logger=False
    )

    # Train the model
    trainer.fit(asr_model)

    # Validate the model
    metrics = trainer.validate(asr_model)

    return metrics[0]['val_epoch_top@1']

# Create and run an Optuna study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=15)

print(f"Best hyperparameters: {study.best_params}")
print(f"Best validation score: {study.best_value}")

# Aggiorna la config con i migliori parametri
cfg.model.optim.lr = study.best_params['learning_rate']
cfg.model.train_ds.batch_size = study.best_params['batch_size']

# Ripeti la procedura per il modello finale
base_jasper = [
    {
        'filters': 64,
        'repeat': 1,
        'kernel': [11],
        'stride': [1],
        'dilation': [1],
        'dropout': 0.3,
        'residual': False
    },
    {
        'filters': 128,
        'repeat': 1,
        'kernel': [13],
        'stride': [1],
        'dilation': [1],
        'dropout': 0.3,
        'residual': True
    }
]

cfg.model.encoder.jasper = base_jasper
num_layers = study.best_params['num_layers']
filters = study.best_params['filters']
kernel_size = study.best_params['kernel_size']
dropout = study.best_params['dropout']

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

final_filters = filters
cfg.model.decoder.feat_in = final_filters

final_model = EncDecClassificationModel(cfg=cfg.model)
final_model.setup_training_data(train_data_config=cfg.model.train_ds)
final_model.setup_validation_data(val_data_config=cfg.model.validation_ds)

final_trainer = pl.Trainer(
    max_epochs=cfg.trainer.max_epochs,
    accelerator=cfg.trainer.accelerator,
    devices=cfg.trainer.devices,
    callbacks=[],
    logger=False
)

final_trainer.fit(final_model)

final_model_path = "../optimized_asr_model.nemo"
final_model.save_to(final_model_path)
print(f"Final model saved at {final_model_path}")

evaluation_script = "evaluate_model.py"
subprocess.run(["python3", evaluation_script], check=True)

# Calcola e stampa il tempo totale
end_time = time.time()
elapsed_time = end_time - start_time
print("---")
print(f"Tempo totale di esecuzione: {elapsed_time:.2f} secondi")