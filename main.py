from model_config import *
from vlqa_data import get_vlqa_data
from data_loader import T5DataLoad
from model import T5Model
from checkpoint_type import *
import pytorch_lightning as pl


if __name__ == "__main__":
    df_train, df_test = get_vlqa_data()
    dataload = T5DataLoad(df_train,df_test)
    dataload.setup()
    device = DEVICE
    model = T5Model()
    model.to(device)
    
    # train
    trainer = pl.Trainer(
        callbacks=[checkpoint_val_loss, checkpoint_f1, checkpoint_em, checkpoint_rouge],
        max_epochs= EPOCHS,
        accelerator="gpu",
        devices = 1,
        check_val_every_n_epoch = 1
    )
    
    trainer.fit(model, dataload)
