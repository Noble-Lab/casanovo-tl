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
from torch.utils.data import Dataset, DataLoader

from depthcharge.transformers import SpectrumTransformerEncoder
from depthcharge.feedforward import FeedForward

import depthcharge as dc
import polars as pl

class EmbDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        print(len(data), len(labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


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

for downsample in ['1024', '512', '256', '128', '64', '32', '16', '8', '4', '2']:
    print("Downsample:", downsample)
    # Open the file for reading in binary mode
    print("Read embeddings")
    with open(f"embeddings/{downsample}/PXD012174_embeddings_w_precursor.pkl", "rb") as f:
        embeds = pickle.load(f)

    print("Read labels")
    with open(f"embeddings/{downsample}/PXD012174_labels.pkl", "rb") as f:
        phospho_labels = pickle.load(f)

    splits = {'a':[],'b':[],'c':[],'d':[]}
    for quart in ordered_ds_names.split("\n")[1:-1]:
        for ds,split_name in zip(quart.split(" "), ['a','b','c','d']):
            splits[split_name].append(ds)
    #Add manually
    splits['d'].append("ACHN")

    train_dataset_names = splits['c'] + splits['d']
    n_train = sum([len(phospho_labels[key]) for key in phospho_labels.keys() if key in train_dataset_names])
    indices = list(range(n_train))
    random.shuffle(indices)

    # Non-pretrained Depthcharge 
    class EmbeddingClassifier(L.LightningModule):
        """A model for clssifying mass spectra by quality."""
        def __init__(self, *args, **kwargs):
            """Initialize the model."""
            super().__init__(*args, **kwargs)
            self.head = torch.nn.Sequential(torch.nn.Linear(514, 512), torch.nn.ReLU(), torch.nn.Linear(512, 1), torch.nn.Sigmoid())
            self.auroc = BinaryAUROC()

        def step(self, batch, step_type):
            """A single step"""
            embs = batch[0].type(torch.float)
            Y = batch[1].type(torch.float)
            Y_hat = self.head(embs).flatten()
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
            embs = batch[0].type(torch.float)
            return self.head(embs).flatten()

        def configure_optimizers(self):
            """Configure optimizers for training."""
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
            return optimizer

    train_data = np.array(list(chain.from_iterable(embeds[key] for key in embeds.keys() if key in train_dataset_names))).astype(np.double)[indices]
    train_labels = np.array(list(chain.from_iterable(phospho_labels[key] for key in phospho_labels.keys() if key in train_dataset_names)))[indices]

    eval_data = np.array(list(chain.from_iterable(embeds[key] for key in embeds.keys() if key in splits['b'])))
    eval_labels = np.array(list(chain.from_iterable(phospho_labels[key] for key in phospho_labels.keys() if key in splits['b'])))

    train_dataset = EmbDataset(train_data, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    valid_dataset = EmbDataset(eval_data, eval_labels)
    valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=True)


    interval = int(len(train_data)/2 - 1)
    print("Non-pretrained Depthcharge training")
    logger = CSVLogger("logs", "model", version=0)
    trainer = L.Trainer(
        max_epochs=1000, 
        val_check_interval=interval,
        callbacks=[L.pytorch.callbacks.early_stopping.EarlyStopping(monitor="valid_auroc", mode="max", patience=5)]
    )
    model = EmbeddingClassifier()
    trainer.fit(model, train_loader, valid_loader)

    print("Non-pretrained Depthcharge inference")
    dc_proba = {}

    test_dataset_names = splits ['a']
    for ds in test_dataset_names:
        test_data = np.array(embeds[ds])
        test_labels = np.array(phospho_labels[ds])

        test_dataset = EmbDataset(test_data, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=128)
            
        pred = trainer.predict(model, test_loader)
        dc_proba[ds] = torch.cat(pred).detach().cpu().numpy()


    print("Saving binned depthcharge predictions to disk")
    with open(f"results/{downsample}/PXD012174_casanovo_predictions.pkl", "wb") as f:
        pickle.dump(dc_proba, f, protocol=pickle.HIGHEST_PROTOCOL)
