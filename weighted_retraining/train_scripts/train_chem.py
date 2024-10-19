""" Script to train chem model """

import argparse
import pytorch_lightning as pl

# My imports
from weighted_retraining.chem.chem_model import JTVAE
from weighted_retraining.chem.chem_data import WeightedJTNNDataset
from weighted_retraining import utils

if __name__ == "__main__":

    # Create arg parser
    parser = argparse.ArgumentParser()
    parser = JTVAE.add_model_specific_args(parser)
    parser = WeightedJTNNDataset.add_model_specific_args(parser)
    parser = utils.DataWeighter.add_weight_args(parser)
    utils.add_default_trainer_args(parser, default_root="logs/train/JTVAE") 

    # Parse arguments
    hparams = parser.parse_args()
    pl.seed_everything(hparams.seed) 

    # Create data
    datamodule = WeightedJTNNDataset(hparams, utils.DataWeighter(hparams), train_frac=1.0)
    datamodule.setup("fit")

    # Load model
    model = JTVAE(hparams, datamodule.vocab)
    if hparams.load_from_checkpoint is not None:
        model = JTVAE.load_from_checkpoint(hparams.load_from_checkpoint, vocab=datamodule.vocab)
        # utils.update_hparams(hparams, model)

    # Main trainer
    trainer = pl.Trainer(
        accelerator='gpu' if hparams.gpu else 'cpu', devices=1,
        # gpus=1 if hparams.gpu else 0,
        default_root_dir=hparams.root_dir,
        max_epochs=hparams.max_epochs,
        callbacks=pl.callbacks.ModelCheckpoint( # checkpoint_callback
            every_n_epochs=1, monitor="loss/val", save_top_k=-1,
            save_last=True
        ),
        detect_anomaly=True, # terminate_on_nan
        gradient_clip_val=20.0   # Model is prone to large gradients
    )


    # Fit
    trainer.fit(model, datamodule=datamodule)
    print("Training finished; end of script")
