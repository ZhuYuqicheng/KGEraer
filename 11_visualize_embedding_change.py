# %%
import argparse
import json
import logging
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy
import time
import pickle
import tqdm
from matplotlib import pyplot as plt 

import torch
import torch.optim

import models
import optimizers.regularizers as regularizers
from datasets.kg_dataset import KGDataset
from models import all_models
from optimizers.kg_optimizer import KGOptimizer
from utils.train import get_savedir, avg_both, format_metrics, count_params, get_khop_entities
from utils.deletion import special_deletion, random_deletion

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
        initial_entity_embedding = initial_model.entity.weight.data.cpu().numpy()
        initial_rel_embedding = initial_model.rel.weight.data.cpu().numpy()
        logging.info(f"----------------------- initial model -----------------------------")
        test_metrics = avg_both(*initial_model.compute_metrics(data["test_examples"], data["filters"]))
        logging.info(format_metrics(test_metrics, split="test"))
        retrained_entity_embeddings = []
        retrained_rel_embeddings = []
        # entity_distances, rel_distances = [], []
        for _ in range(5):
            # model = unlearner.train_model(args, "finetune", initial_model=initial_model)
            model = unlearner.train_model(args, "finetune", initial_model=initial_model, fix_rel=True)
            logging.info(f"----------------------------------------------------")
            test_metrics = avg_both(*model.compute_metrics(data["test_examples"], data["filters"]))
            logging.info(format_metrics(test_metrics, split="test"))
            retrained_entity_embedding = model.entity.weight.data.cpu().numpy()
            retrained_rel_embedding = model.rel.weight.data.cpu().numpy()
            retrained_entity_embeddings.append(retrained_entity_embedding)
            retrained_rel_embeddings.append(retrained_rel_embedding)
            # distance evaluation
            # entity_distances.append(torch.sum(torch.square(initial_model.entity.weight.data - model.entity.weight.data), dim=1).cpu())
            # rel_distances.append(torch.sum(torch.square(initial_model.rel.weight.data - model.rel.weight.data), dim=1).cpu())
        # logging.info(f"----------------------- distance evaluation -----------------------------")
        # logging.info(f"")
        
        return initial_entity_embedding, initial_rel_embedding, retrained_entity_embeddings, retrained_rel_embeddings, data["deleted_triples"]

if __name__ == "__main__":
    pwd = os.getcwd()
    os.environ["LOG_DIR"] = f"{pwd}/logs"
    os.environ["DATA_PATH"] = f"{pwd}/data"
    args = argparse.Namespace(
        dataset = "Nations", # "FB15K", "WN", "WN18RR", "FB237", "YAGO3-10"
        model = "TransE", # TransE, RotE; ComplEx, RotatE
        regularizer = "N3", # N3 or F2
        reg = .1, 
        optimizer = "Adam", 
        max_epochs = 500, 
        patience = 500, 
        valid = 500,
        rank = 2, 
        batch_size = 1000, 
        neg_sample_size = -1, 
        dropout = 0, 
        init_size = 1e-3, 
        learning_rate = 0.001, 
        gamma = 1,
        bias = "constant",
        dtype = "double",
        double_neg = True, 
        debug = False, 
        multi_c = False,
        device = "cuda:2",
        random_seed = 3
    )

    initial_entity_embedding, initial_rel_embedding, retrained_entity_embeddings, retrained_rel_embeddings, deleted_triple = \
        unlearning(args, "random", 1)

# 
entity_list = ["egypt","china","cuba","netherlands","india","usa","jordan","burma","brazil","indonesia","poland","uk","ussr","israel"]
entity_df = pd.read_pickle("/workspace/LLKGE/KGEmb/data/Nations/entity_dist.pkl")
entity_sizes = entity_df.sort_values(by="entities")["count"].to_list()
rel_df = pd.read_pickle("/workspace/LLKGE/KGEmb/data/Nations/relation_dist.pkl")
rel_sizes = rel_df.sort_values(by="relation")["count"].to_list()
deleted_entities = [entity_list[deleted_triple.numpy()[0,0]], entity_list[deleted_triple.numpy()[0,2]]]

plt.figure(figsize=(12, 10))
plt.scatter(initial_entity_embedding[:,0], initial_entity_embedding[:,1], s=entity_sizes)
# plt.scatter(initial_entity_embedding[:,0], initial_entity_embedding[:,1])
for i, entity in enumerate(entity_list):
    plt.annotate(entity, (initial_entity_embedding[i,0], initial_entity_embedding[i,1]), textcoords="offset points", xytext=(0,0), ha='center', fontsize=14)
plt.annotate(deleted_entities[0], (initial_entity_embedding[deleted_triple.numpy()[0,0],0], initial_entity_embedding[deleted_triple.numpy()[0,0],1]), 
             textcoords="offset points", xytext=(0,0), ha='center', color="red", fontsize=14)
plt.annotate(deleted_entities[1], (initial_entity_embedding[deleted_triple.numpy()[0,2],0], initial_entity_embedding[deleted_triple.numpy()[0,2],1]), 
             textcoords="offset points", xytext=(0,0), ha='center', color="red", fontsize=14)
for show_ind in range(5):
    plt.scatter(retrained_entity_embeddings[show_ind][:,0], retrained_entity_embeddings[show_ind][:,1])
    # draw arrows
    for i in range(len(initial_entity_embedding)):
        x1, y1 = initial_entity_embedding[i]
        x2, y2 = retrained_entity_embeddings[show_ind][i]
        dx = x2 - x1
        dy = y2 - y1
        plt.arrow(x1, y1, dx, dy, width=1e-5, head_width=2e-4, color="grey", length_includes_head=True)

plt.show()

# 
plt.figure(figsize=(12, 10))
plt.scatter(initial_rel_embedding[:,0], initial_rel_embedding[:,1])
for show_ind in range(5):
    plt.scatter(retrained_rel_embeddings[show_ind][:,0], retrained_rel_embeddings[show_ind][:,1])
    # draw arrows
    for i in range(len(initial_rel_embedding)):
        x1, y1 = initial_rel_embedding[i]
        x2, y2 = retrained_rel_embeddings[show_ind][i]
        dx = x2 - x1
        dy = y2 - y1
        plt.arrow(x1, y1, dx, dy, width=1e-5, head_width=2e-4, color="grey", length_includes_head=True)

plt.show()
# %%
plt.figure(figsize=(8, 6))
ind = 1
plt.scatter(initial_embedding[ind,0], initial_embedding[ind,1])
for embed in retrained_embeddings:
    plt.scatter(embed[ind,0], embed[ind,1])
# %%
