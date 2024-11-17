import os
import optuna
import pytorch_lightning as pl
from omegaconf import OmegaConf
from nemo.collections.asr.models import EncDecClassificationModel
from nemo.utils.exp_manager import exp_manager
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch

def objective(trial):
    # Carica la configurazione
    cfg = OmegaConf.load("/home/giolinux/NLP_Project/config.yaml")

    # Imposta il seed per la riproducibilità
    pl.seed_everything(42)

    # Scegli gli iperparametri da ottimizzare
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    dropout = trial.suggest_uniform('dropout', 0.1, 0.5)
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16, 32])
    filters = trial.suggest_categorical('filters', [64, 128, 256])
    num_layers = trial.suggest_int('num_layers', 2, 5)
    kernel_size = trial.suggest_int('kernel_size', 3, 11, step=2)
    activation = trial.suggest_categorical('activation', ['relu', 'selu', 'gelu'])
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
    grad_clip = trial.suggest_uniform('grad_clip', 0.0, 1.0)
    optimizer = trial.suggest_categorical('optimizer', ['adam', 'sgd', 'adamw'])
    scheduler = trial.suggest_categorical('scheduler', ['CosineAnnealing', 'StepLR', 'ExponentialLR'])
    warmup_steps = trial.suggest_int('warmup_steps', 0, 1000)
    label_smoothing = trial.suggest_uniform('label_smoothing', 0.0, 0.1)
    bn_momentum = trial.suggest_uniform('bn_momentum', 0.0, 1.0)

    # Aggiorna la configurazione con gli iperparametri scelti
    cfg.model.optim.lr = lr
    cfg.model.optim.weight_decay = weight_decay
    cfg.model.optim.name = optimizer
    cfg.model.optim.sched = {'name': scheduler, 'warmup_steps': warmup_steps}

    cfg.trainer.gradient_clip_val = grad_clip

    cfg.model.train_ds.batch_size = batch_size
    cfg.model.validation_ds.batch_size = batch_size

    # Aggiorna l'encoder con il numero di layer e altri parametri
    cfg.model.encoder.jasper = []
    for i in range(num_layers):
        layer = {
            'filters': filters,
            'repeat': 1,
            'kernel': [kernel_size],
            'stride': [1],
            'dilation': [1],
            'dropout': dropout,
            'residual': True if i > 0 else False,
            'activation': activation,
            'norm_kwargs': {'momentum': bn_momentum}
        }
        cfg.model.encoder.jasper.append(layer)
    # Aggiorna il feat_in dell'encoder e del decoder
    cfg.model.encoder.feat_in = cfg.model.preprocessor.features
    cfg.model.decoder.feat_in = filters

    # Aggiungi label smoothing se supportato
    cfg.model.loss = {'label_smoothing': label_smoothing}

    # Disabilita il checkpointing e il logging per velocizzare l'ottimizzazione
    cfg.trainer.enable_checkpointing = False
    cfg.trainer.logger = False
    cfg.exp_manager.create_checkpoint_callback = False
    cfg.exp_manager.create_tensorboard_logger = False

    # Imposta il numero di epoch per una valutazione rapida
    cfg.trainer.max_epochs = 10  # Puoi aumentare questo valore se hai più tempo

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
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        logger=False,
        enable_checkpointing=False,
        callbacks=[early_stop_callback],
        precision=16,  # Abilita la precisione mista per accelerare l'addestramento
        gradient_clip_val=grad_clip,
    )

    # Configura l'exp_manager
    exp_manager(trainer, cfg.exp_manager)

    # Inizializza il modello
    asr_model = EncDecClassificationModel(cfg=cfg.model)

    # Imposta i dati
    asr_model.setup_training_data(train_data_config=cfg.model.train_ds)
    asr_model.setup_validation_data(val_data_config=cfg.model.validation_ds)

    # Aggiungi un try-except per gestire eventuali errori
    try:
        # Avvia l'addestramento
        trainer.fit(asr_model)
    except Exception as e:
        print(f"Errore durante l'addestramento: {e}")
        return float('inf')

    # Valuta il modello
    result = trainer.validate(asr_model)

    # Ottieni la metrica da ottimizzare (ad esempio, val_loss)
    val_loss = result[0]['val_loss']

    # Registra la metrica intermedia per Optuna
    trial.report(val_loss, step=cfg.trainer.max_epochs)

    # Pruning
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    # Restituisci la metrica per l'ottimizzazione
    return val_loss

if __name__ == '__main__':
    # Imposta la precisione delle moltiplicazioni
    torch.set_float32_matmul_precision('high')

    # Specifica il percorso per il database SQLite
    storage_name = "sqlite:///asr_study.db"

    # Crea uno studio Optuna con storage persistente e pruner
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(
        direction='minimize',
        study_name='ASR_Hyperparameter_Optimization',
        storage=storage_name,
        load_if_exists=True,
        pruner=pruner
    )

    # Esegui l'ottimizzazione
    study.optimize(objective, n_trials=10)

    # Stampa i migliori iperparametri trovati
    print("I migliori iperparametri trovati sono:")
    print(study.best_params)
