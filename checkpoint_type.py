from pytorch_lightning.callbacks import ModelCheckpoint

from model_config import *

# ModelCheckpoint cho F1
checkpoint_val_loss = ModelCheckpoint(
    dirpath=OUTPUT_DIR,
    filename='best-model-loss',
    save_top_k=1,
    verbose=True,
    monitor="val_loss",
    mode="min"
)

# ModelCheckpoint cho F1
checkpoint_f1 = ModelCheckpoint(
    dirpath=OUTPUT_DIR,
    filename='best-model-f1',
    save_top_k=1,
    verbose=True,
    monitor="val_f1",
    mode="max"
)

# ModelCheckpoint cho EM
checkpoint_em = ModelCheckpoint(
    dirpath=OUTPUT_DIR,
    filename='best-model-em',
    save_top_k=1,
    verbose=True,
    monitor="val_em",
    mode="max"
)

# ModelCheckpoint cho EM
checkpoint_rouge = ModelCheckpoint(
    dirpath=OUTPUT_DIR,
    filename='best-model-rouge',
    save_top_k=1,
    verbose=True,
    monitor="val_rouge",
    mode="max"
)