# %%
import argparse
import logging
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import torch
import torch.optim
from utils.train import get_savedir
from KGUnlearn import DataLoader, UnlearnPipeline

def unlearning(args, deletion_mode, n_del, degree_set=None, repeat=1, save_model=False):
    save_dir = get_savedir(args.model, args.dataset)

    # file logger
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=os.path.join(save_dir, "train.log")
    )

    # stdout logger
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)
    logging.info("Saving logs in: {}".format(save_dir))

    # set random seed
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    for i in range(repeat):
        logging.info(f"--------------------------------------------------------------")
        logging.info(f"Training {i+1}/{repeat}-th {n_del}-{deletion_mode} deletion...")
        if degree_set is not None: logging.info(f"Degree setting: {degree_set}")

        model_dir = os.path.join(save_dir, str(n_del)+"_"+deletion_mode+"_"+str(i+1))
        os.makedirs(model_dir)

        # load data
        data_loader = DataLoader(args)
        data = data_loader.load_data()
        if deletion_mode == "random":
            data["deleted_triples"], data["del_train_examples"] = data_loader.random_deletion(n_del=n_del)
        elif deletion_mode == "degree":
            data["deleted_triples"], data["del_train_examples"] = data_loader.degree_deletion(degree_set["entity"], degree_set["relation"], n_del)
        
        if save_model: torch.save(data["deleted_triples"], os.path.join(model_dir, "deleted_triples.pt"))

        # run unlearning training
        unlearner = UnlearnPipeline(args, model_dir, console, data)
        initial_model = unlearner.train_model(args, "initial")

        return initial_model

if __name__ == "__main__":
    pwd = os.getcwd()
    os.environ["LOG_DIR"] = f"{pwd}/logs"
    os.environ["DATA_PATH"] = f"{pwd}/data"
    args = argparse.Namespace(
        dataset = "Nations", # "FB15K", "WN", "WN18RR", "FB237", "YAGO3-10"
        model = "TransE", # TransE, RotE; ComplEx, RotatE
        regularizer = "N3",
        reg = 0, 
        optimizer = "Adam", 
        max_epochs = 50, 
        patience = 10, 
        valid = 2,
        rank = 2, 
        batch_size = 1000, 
        neg_sample_size = 50, 
        dropout = 0, 
        init_size = 1e-3, 
        learning_rate = 0.005, 
        gamma = 1,
        bias = "constant",
        dtype = "double",
        double_neg = True, 
        debug = False, 
        multi_c = False,
        device = "cuda:2",
        random_seed = 0
    )

    unlearning(args, deletion_mode, n_del, degree_set=None, repeat=1, save_model=False)