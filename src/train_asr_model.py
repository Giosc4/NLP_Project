import os
import pytorch_lightning as pl
from omegaconf import OmegaConf
import nemo.collections.asr as nemo_asr
from nemo.utils.exp_manager import exp_manager

# Load the configuration
cfg = OmegaConf.load("config.yaml")

# Initialize the Trainer
trainer = pl.Trainer(**cfg.trainer)

# Initialize the Model
asr_model = nemo_asr.models.EncDecClassificationModel(cfg=cfg.model, trainer=trainer)

# Setup experiment manager
exp_manager(trainer, cfg.get("exp_manager", None))

# Start training
trainer.fit(asr_model)

# Save the trained model
asr_model.save_to("asr_model.nemo")

print("Training completed and model saved.")
