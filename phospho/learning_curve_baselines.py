import os, random
import contextlib
import gc
from itertools import chain
import pickle

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import ppx
import pandas as pd
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score

import torch
from torch.utils.data import DataLoader
from xgboost import XGBClassifier
from zipfile import ZipFile as zipfile

import lightning as L
import torch.nn.functional as F
from lightning.pytorch.loggers import CSVLogger
from torchmetrics.classification import BinaryAUROC

from depthcharge.transformers import SpectrumTransformerEncoder
from depthcharge.feedforward import FeedForward

import depthcharge as dc
import polars as pl

#Copied from Supp Table 2 in AHLF paper. Missing ACHN in d split.
ordered_ds_names = """
Primary-AML BoneMarrow SW480 MCF7
Daudi A673 Lung Kit255
U2OS Mono-Mac-1 MV-4-11 SKM-1
Primary-Gastro WM239A 143B.TK BT474
HeLa Primary-Ovarian Jurkat HCT116
HEK293 A431 DG75 M019i
A549 Fibroblast U937 Primary-Pancreas
HaCaT HUVEC H1975 Metastasis-Pancreas
OVAS KOC-7C HCCLM6 OVISE
Primary-Prostate MCF10A Primary-Melanoma SH-SY5Y
TOV-21-Primary Primary-BOEC Primary-Colorectal Platelets
LNCaP MDA-MB-231 H3255 11-18
HEPG2 Primary-Breast-MicroAndExo H1299 SK-N-BE
Colon Primary-Glioblastoma RKO Brain
Kasumi-1 Muscle COLO-205 P31.Fuj
HT-29 SW1398 THP1 ECV-304
ES2-Primary DLD-1 CACO-2 GB2
OVSAYO H358 Primary-Glioma Primary-Liver
HDMVEC DoHH2 SU-DHL-6 RL
PANC-05-04 CTS PC9 Capan-1
SU.86.86 HPAF-II BxPC-3 PANC-04-03
CFPAC-1 PANC-08-13 Capan-2 SW1990
RPMI-8226 Hs-700-T PANC-10-05 PL45
HPAC Hs-766-T PANC-03-27 OMP2
PANC-02-03 AsPC-1 CL1-0 U266B1
"""

for downsample in ['1024', '512', '256', '128', '64', '32', '16', '8', '4', '2', '1']:
    print('downsample')
    splits = {'a':[],'b':[],'c':[],'d':[]}
    for quart in ordered_ds_names.split("\n")[1:-1]:
        for ds,split_name in zip(quart.split(" "), ['a','b','c','d']):
            splits[split_name].append(ds)
    #Add manually
    splits['d'].append("ACHN")


    train_dataset_names = splits['c'] + splits['d']
    valid_dataset_names = splits['b']
    test_dataset_names = splits ['a']

    mgf_dfs = {}

    for sp in ['a','b','c','d']:
        print(f"Split {sp}")
        print(splits[sp])
        for ds in splits[sp]:
            phos_df = dc.data.spectra_to_df(f'data/{downsample}/{ds}/phospho.mgf', ms_level = [2], progress=True)
            non_phos_df = dc.data.spectra_to_df(f'data/{downsample}/{ds}/non_phospho.mgf', ms_level = [2], progress=True)
            mgf_dfs[ds] = pl.concat([phos_df, non_phos_df], how="vertical").with_columns(pl.Series(name="label", values=len(phos_df) * [1] + len(non_phos_df) * [0]))

    train_data = dc.data.SpectrumDataset(pl.concat([mgf_dfs[ds] for ds in train_dataset_names], how="vertical").sample(fraction=1.0, shuffle=True) ,batch_size=32)
    valid_data = dc.data.SpectrumDataset(pl.concat([mgf_dfs[ds] for ds in valid_dataset_names], how="vertical").sample(fraction=1.0, shuffle=True) ,batch_size=32)

    train_data.samples = 0
    train_data.batch_size = 32

    valid_data.samples = 1_000_000
    valid_data.batch_size = 128

    train_loader = DataLoader(train_data, batch_size=None)
    valid_loader = DataLoader(valid_data, batch_size=None)

    # Train binned embeddings baseline
    print("Train binned embeddings baseline")
    def bin_spectra(batch, n_bins=100, min_mz=140, max_mz=2000):
        """Bin mass spectra.

        Parameters
        ----------
        batch : dict of torch.Tensor
            The batch of data.
        n_bins : int
            The number of bins.
        min_mz : float
            The lowest m/z bin.
        max_mz : float
            The highest m/z bin.
    
        Returns
        -------
        torch.Tensor
            The batch of binned mass spectra.
        """
        bins = torch.linspace(min_mz, max_mz, n_bins - 1)
        out = torch.empty(batch["mz_array"].shape[0], n_bins)
        binned = torch.bucketize(batch["mz_array"], bins)
        for i in range(out.shape[0]):
            out[i, :] = torch.bincount(
                binned[i, :], 
                weights=batch["intensity_array"][i, :], 
                minlength=n_bins,
            )

        return out

    X_train, y_train = zip(*[(bin_spectra(b), b["label"]) for b in train_loader])
    X_train = torch.vstack(X_train).detach().cpu().numpy()
    y_train = torch.cat(y_train).detach().cpu().numpy()

    X_eval, y_eval = zip(*[(bin_spectra(b), b["label"]) for b in valid_loader])
    X_eval = torch.vstack(X_eval).detach().cpu().numpy()
    y_eval = torch.cat(y_eval).detach().cpu().numpy()

    xgb = XGBClassifier(eval_metric='auc', early_stopping_rounds=32,n_estimators=5000).fit(X_train, y_train, eval_set=[(X_eval, y_eval)])

    # Binned embeddings inference
    print("Binned embeddings inference")
    baseline_proba = {}

    test_labels = {}
    for ds in test_dataset_names:
        test_data = dc.data.SpectrumDataset(mgf_dfs[ds], batch_size=32)

        test_data.samples = 0
        test_data.batch_size = 128

        test_loader = DataLoader(test_data, batch_size=None)
            
        
        X_test, y_test = zip(*[(bin_spectra(b), b["label"]) for b in test_loader])

        X_test = torch.vstack(X_test).detach().cpu().numpy()
        y_test = torch.cat(y_test).detach().cpu().numpy()
        
        baseline_proba[ds] = xgb.predict_proba(X_test)[:, 1]
        test_labels[ds] = y_test


    print("Saving binned baseline predictions to disk")
    with open(f"results/{downsample}/PXD012174_binned_baseline_predictions.pkl", "wb") as f:
        pickle.dump(baseline_proba, f, protocol=pickle.HIGHEST_PROTOCOL)


    print("Saving labels to disk")
    with open(f"results/{downsample}/small_labels.pkl", "wb") as f:
        pickle.dump(test_labels, f, protocol=pickle.HIGHEST_PROTOCOL)


    # Non-pretrained Depthcharge 
    class SpectrumClassifier(SpectrumTransformerEncoder, L.LightningModule):
        """A model for clssifying mass spectra by quality."""
        def __init__(self, *args, **kwargs):
            """Initialize the model."""
            super().__init__(*args, **kwargs)
            self.mz_encoder = dc.encoders.FloatEncoder(self.d_model)
            self.charge_encoder = dc.encoders.FloatEncoder(self.d_model, 1, 10)
            self.head = torch.nn.Sequential(torch.nn.Linear(self.d_model, 1), torch.nn.Sigmoid())
            self.auroc = BinaryAUROC()

        def global_token_hook(self, mz_array, precursor_mz, precursor_charge, *args, **kwargs):
            """Use our cls token"""
            mz_emb = self.mz_encoder(precursor_mz[:, None])
            charge_emb = self.charge_encoder(precursor_charge.type_as(mz_array)[:, None])
            return (mz_emb + charge_emb).squeeze()

        def step(self, batch, step_type):
            """A single step"""
            Y = batch["label"].type_as(batch["mz_array"])
            Y_hat = self.head(self(**batch)[0][:, 0, :]).flatten()
            loss = F.binary_cross_entropy(Y_hat, Y)
            self.log(f"{step_type}_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True)
            if step_type == "valid":
                self.auroc(Y_hat, Y)
                self.log(f"{step_type}_auroc", self.auroc, on_step=False, on_epoch=True, prog_bar=True)
            return loss
        
        def training_step(self, batch, batch_idx):
            """The training step."""
            return self.step(batch, "train")

        def validation_step(self, batch, batch_idx):
            """The validation step."""
            return self.step(batch, "valid")

        def predict_step(self, batch, batch_idx):
            """The predict step."""
            return self.head(self(**batch)[0][:, 0, :]).flatten()

        def configure_optimizers(self):
            """Configure optimizers for training."""
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-6)
            return optimizer
    
    interval = int(len(train_data)/2 - 1)
    print("Non-pretrained Depthcharge training")
    logger = CSVLogger("logs", "model", version=0)
    trainer = L.Trainer(
        max_epochs=1000, 
        val_check_interval=interval,
        callbacks=[L.pytorch.callbacks.early_stopping.EarlyStopping(monitor="valid_auroc", mode="max", patience=5)]
    )

    model = SpectrumClassifier(d_model=512, nhead=16, n_layers=3) # peak_encoder=enc
    trainer.fit(model, train_loader, valid_loader)

    # Non-pretrained Depthcharge inference
    print("Non-pretrained Depthcharge inference")
    dc_proba = {}

    for ds in test_dataset_names:
        print(f"{ds} contains {len(mgf_dfs[ds])} spectra")
        test_data = dc.data.SpectrumDataset(mgf_dfs[ds], batch_size=16)
        test_data.samples = 0
        test_data.batch_size = 512

        test_loader = DataLoader(test_data, batch_size=None)
            
        test_data.samples = 0
        pred = trainer.predict(model, test_loader)
        dc_proba[ds] = torch.cat(pred).detach().cpu().numpy()


    print("Saving binned depthcharge predictions to disk")
    with open(f"results/{downsample}/PXD012174_depthcharge_predictions.pkl", "wb") as f:
        pickle.dump(dc_proba, f, protocol=pickle.HIGHEST_PROTOCOL)
        

    