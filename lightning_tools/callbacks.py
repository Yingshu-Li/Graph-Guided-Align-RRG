import os
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

def add_callbacks(args):
    log_dir = args.savedmodel_path
    os.makedirs(log_dir, exist_ok=True)

    # --------- Add Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(log_dir, "checkpoints_state"),
        filename='{epoch:d}',
        save_top_k=1,
        every_n_train_steps=args.every_n_train_steps,  # no save
        # every_n_epochs=1,
        save_last=False,
        save_weights_only=False,
        save_on_train_epoch_end = True
    )
    
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.path.join(log_dir, "logs"), name="tensorboard")
    csv_logger = CSVLogger(save_dir=os.path.join(log_dir, "logs"), name="csvlog")
    if args.use_wandb:
        wandb_logger = WandbLogger(project=args.project_name, save_dir=os.path.join(log_dir, "logs"), name=args.version, entity="usyd_med")

        to_returns = {
            "callbacks": [checkpoint_callback, lr_monitor_callback],
            "loggers": [csv_logger, tb_logger, wandb_logger]
        }
    else:
        to_returns = {
            "callbacks": [checkpoint_callback, lr_monitor_callback],
            "loggers": [csv_logger, tb_logger]
        }
    return to_returns
