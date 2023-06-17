#%%
"""Analyse the global embedding change"""

import argparse
import json
import logging
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy

import torch
import torch.optim

import models
import optimizers.regularizers as regularizers
from datasets.kg_dataset import KGDataset
from models import all_models
from optimizers.kg_optimizer import KGOptimizer
from utils.train import get_savedir, avg_both, format_metrics, count_params, get_khop_entities
from utils.deletion import special_deletion, random_deletion

def initial_train(args, deleted_triples=None, eval_train=False):
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    # create dataset
    dataset_path = os.path.join(os.environ["DATA_PATH"], args.dataset)
    dataset = KGDataset(dataset_path, args.debug)
    args.sizes = dataset.get_shape()

    # load data
    logging.info("\t " + str(dataset.get_shape()))
    if deleted_triples is not None:
        train_examples = dataset.get_examples("train", del_triple=deleted_triples)
    else:
        train_examples = dataset.get_examples("train")
    valid_examples = dataset.get_examples("valid")
    test_examples = dataset.get_examples("test")
    filters = dataset.get_filters()

    # save config
    with open(os.path.join(args.save_dir, "config.json"), "w") as fjson:
        json.dump(vars(args), fjson)

    # create model
    model = getattr(models, args.model)(args)
    total = count_params(model)
    logging.info("Total number of parameters {}".format(total))
    # device = "cuda"
    model.to(args.device)

    # get optimizer
    regularizer = getattr(regularizers, args.regularizer)(args.reg)
    optim_method = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.learning_rate)
    optimizer = KGOptimizer(model, regularizer, optim_method, args.batch_size, args.neg_sample_size,
                            bool(args.double_neg), args.device)

    counter = 0
    best_mrr = None
    best_epoch = None
    logging.info("\t Start training")
    for step in range(args.max_epochs):

        # Train step
        model.train()
        train_loss = optimizer.epoch(train_examples)
        logging.info("\t Epoch {} | average train loss: {:.4f}".format(step, train_loss))

        # Valid step
        model.eval()
        valid_loss = optimizer.calculate_valid_loss(valid_examples)
        logging.info("\t Epoch {} | average valid loss: {:.4f}".format(step, valid_loss))

        if (step + 1) % args.valid == 0:
            valid_metrics = avg_both(*model.compute_metrics(valid_examples, filters))
            logging.info(format_metrics(valid_metrics, split="valid"))

            valid_mrr = valid_metrics["MRR"]
            if not best_mrr or valid_mrr > best_mrr:
                best_mrr = valid_mrr
                counter = 0
                best_epoch = step
                logging.info("\t Saving model at epoch {} in {}".format(step, args.save_dir))
                torch.save(model.cpu().state_dict(), os.path.join(args.save_dir, "model.pt"))
                model.to(args.device)
            else:
                counter += 1
                if counter == args.patience:
                    logging.info("\t Early stopping")
                    break
                elif counter == args.patience // 2:
                    pass

    logging.info("\t Optimization finished")
    if not best_mrr:
        torch.save(model.cpu().state_dict(), os.path.join(args.save_dir, "model.pt"))
    else:
        logging.info("\t Loading best model saved at epoch {}".format(best_epoch))
        model.load_state_dict(torch.load(os.path.join(args.save_dir, "model.pt")))
    model.to(args.device)
    model.eval()

    # Training metrics
    if eval_train:
        eval_train_examples = train_examples[train_examples[:,1]<args.sizes[1]//2]
        train_metrics = avg_both(*model.compute_metrics(eval_train_examples, filters))
        logging.info(format_metrics(train_metrics, split="train"))
    # Validation metrics
    valid_metrics = avg_both(*model.compute_metrics(valid_examples, filters))
    logging.info(format_metrics(valid_metrics, split="valid"))
    # Test metrics
    test_metrics = avg_both(*model.compute_metrics(test_examples, filters))
    logging.info(format_metrics(test_metrics, split="test"))

    pos_ranks, neg_ranks, pos_plausibility, neg_plausibility \
        = model.compute_unlearning_metrics(train_examples, filters, deleted_triples)
    
    unlearn_metrics = {"pos_ranks": pos_ranks, "neg_ranks": neg_ranks, "pos_plausibility": pos_plausibility, "neg_plausibility": neg_plausibility}

    return model, unlearn_metrics

def fine_tune(args, old_model, deleted_triples=None, eval_train=False):
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    # create dataset
    dataset_path = os.path.join(os.environ["DATA_PATH"], args.dataset)
    dataset = KGDataset(dataset_path, args.debug)
    args.sizes = dataset.get_shape()

    # load data
    logging.info("\t " + str(dataset.get_shape()))
    if deleted_triples is not None:
        train_examples = dataset.get_examples("train", del_triple=deleted_triples)
    else:
        train_examples = dataset.get_examples("train")
    valid_examples = dataset.get_examples("valid")
    test_examples = dataset.get_examples("test")
    filters = dataset.get_filters()

    model = copy.deepcopy(old_model)

    # get optimizer
    regularizer = getattr(regularizers, args.regularizer)(args.reg)
    optim_method = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.learning_rate)
    optimizer = KGOptimizer(model, regularizer, optim_method, args.batch_size, args.neg_sample_size,
                            bool(args.double_neg), args.device)
    
    for step in range(args.max_epochs):
        # Train step
        model.train()
        train_loss = optimizer.epoch(train_examples)
        logging.info("\t Epoch {} | average train loss: {:.4f}".format(step, train_loss))
    model.eval()
    # Training metrics
    if eval_train:
        eval_train_examples = train_examples[train_examples[:,1]<args.sizes[1]//2]
        train_metrics = avg_both(*model.compute_metrics(eval_train_examples, filters))
        logging.info(format_metrics(train_metrics, split="train"))
    # Validation metrics
    valid_metrics = avg_both(*model.compute_metrics(valid_examples, filters))
    logging.info(format_metrics(valid_metrics, split="valid"))
    # Test metrics
    test_metrics = avg_both(*model.compute_metrics(test_examples, filters))
    logging.info(format_metrics(test_metrics, split="test"))

def restrict_fine_tune(args, old_model, deleted_triples=None, eval_train=False, fix_rel=False, fix_ent=False, hop=0):
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    # create dataset
    dataset_path = os.path.join(os.environ["DATA_PATH"], args.dataset)
    dataset = KGDataset(dataset_path, args.debug)
    args.sizes = dataset.get_shape()

    # load data
    logging.info("\t " + str(dataset.get_shape()))
    if deleted_triples is not None:
        train_examples = dataset.get_examples("train", del_triple=deleted_triples)
    else:
        train_examples = dataset.get_examples("train")
    valid_examples = dataset.get_examples("valid")
    test_examples = dataset.get_examples("test")
    filters = dataset.get_filters()

    model = copy.deepcopy(old_model)

    # get optimizer
    regularizer = getattr(regularizers, args.regularizer)(args.reg)
    optim_method = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.learning_rate)
    optimizer = KGOptimizer(model, regularizer, optim_method, args.batch_size, args.neg_sample_size,
                            bool(args.double_neg), args.device, deleted_triple=deleted_triples)
    
    if fix_ent: # get freeze_index
        all_index = list(range(model.entity.weight.size()[0]))
        hop_entities = get_khop_entities(train_examples, deleted_triples, hop)
        freeze_index = [index for index in all_index if index not in hop_entities]
    else:
        freeze_index = []

    for step in range(args.max_epochs):
        # Train step
        model.train()
        train_loss = optimizer.restrict_epoch(train_examples, freeze_index, fix_rel=False)
        logging.info("\t Epoch {} | average train loss: {:.4f}".format(step, train_loss))

    # evaluation
    model.eval()
    # Training metrics
    if eval_train:
        eval_train_examples = train_examples[train_examples[:,1]<args.sizes[1]//2]
        train_metrics = avg_both(*model.compute_metrics(eval_train_examples, filters))
        logging.info(format_metrics(train_metrics, split="train"))
    # Validation metrics
    valid_metrics = avg_both(*model.compute_metrics(valid_examples, filters))
    logging.info(format_metrics(valid_metrics, split="valid"))
    # Test metrics
    test_metrics = avg_both(*model.compute_metrics(test_examples, filters))
    logging.info(format_metrics(test_metrics, split="test"))

    pos_ranks, neg_ranks, pos_plausibility, neg_plausibility \
        = model.compute_unlearning_metrics(train_examples, filters, deleted_triples)
    
    unlearn_metrics = {"pos_ranks": pos_ranks, "neg_ranks": neg_ranks, "pos_plausibility": pos_plausibility, "neg_plausibility": neg_plausibility}

    return model, unlearn_metrics

def train(args):
    # args.save_dir = get_savedir(args.model, args.dataset)

    # # file logger
    # logging.basicConfig(
    #     format="%(asctime)s %(levelname)-8s %(message)s",
    #     level=logging.INFO,
    #     datefmt="%Y-%m-%d %H:%M:%S",
    #     filename=os.path.join(args.save_dir, "train.log")
    # )

    # # stdout logger
    # console = logging.StreamHandler()
    # console.setLevel(logging.INFO)
    # formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    # console.setFormatter(formatter)
    # logging.getLogger("").addHandler(console)
    # logging.info("Saving logs in: {}".format(args.save_dir))

    dataset_path = os.path.join(os.environ["DATA_PATH"], args.dataset)
    dataset = KGDataset(dataset_path, args.debug)
    initial_train_examples = dataset.get_examples("train")
    deleted_triples = random_deletion(initial_train_examples, n_del=10)

    # retraining
    # initial_train_examples, _ = initial_train(args, eval_train=False)
    # deleted_triples = random_deletion(initial_train_examples, n_del=1)
    # logging.info("retraining...")
    # initial_train_examples, _ = initial_train(args, eval_train=False, deleted_triples=deleted_triples)

    # fine-tuning
    # initial_train_examples, initial_model = initial_train(args, eval_train=False)
    # deleted_triples = random_deletion(initial_train_examples, n_del=1)
    # logging.info("Fine-tuning...")
    # fine_tune(args, initial_model, deleted_triples=deleted_triples, eval_train=False)

    # restrict fine-tuning
    initial_model, initial_unlearn_metrics = initial_train(args, deleted_triples=deleted_triples, eval_train=False)
    logging.info("Fine-tuning...")
    model, unlearn_metrics = restrict_fine_tune(args, initial_model, deleted_triples=deleted_triples, eval_train=False, fix_rel=False, fix_ent=False, hop=1)

    return deleted_triples, initial_model, model, initial_unlearn_metrics, unlearn_metrics

#%%
if __name__ == "__main__":
    pwd = os.getcwd()
    os.environ["LOG_DIR"] = f"{pwd}/logs"
    os.environ["DATA_PATH"] = f"{pwd}/data"
    args = argparse.Namespace(
        dataset = "WN18RR", # "FB15K", "WN", "WN18RR", "FB237", "YAGO3-10"
        model = "TransE", # TransE, RotE; ComplEx, RotatE
        regularizer = "N3",
        reg = 0, 
        optimizer = "Adam", 
        max_epochs = 50, 
        patience = 10, 
        valid = 3,
        rank = 20, 
        batch_size = 1000, 
        neg_sample_size = 50, 
        dropout = 0, 
        init_size = 1e-3, 
        learning_rate = 0.01, 
        gamma = 2,
        bias = "constant",
        dtype = "double",
        double_neg = True, 
        debug = False, 
        multi_c = False,
        device = "cuda:1",
        random_seed = 42
    )

    args.save_dir = get_savedir(args.model, args.dataset)
    # file logger
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=os.path.join(args.save_dir, "train.log")
    )

    # stdout logger
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)
    logging.info("Saving logs in: {}".format(args.save_dir))

    # retrain_entity_e, retrain_relation_e, del_triple = retrain_from_scratch(args)
    deleted_triples, initial_model, model, initial_unlearn_metrics, unlearn_metrics = train(args)

    pos_delta_rank = torch.mean(torch.abs(initial_unlearn_metrics["pos_ranks"] - unlearn_metrics["pos_ranks"]))
    neg_delta_rank = torch.mean(torch.abs(initial_unlearn_metrics["neg_ranks"] - unlearn_metrics["neg_ranks"]))
    pos_delta_p = torch.mean(torch.abs(initial_unlearn_metrics["pos_plausibility"] - unlearn_metrics["pos_plausibility"]))
    neg_delta_p = torch.mean(torch.abs(initial_unlearn_metrics["neg_plausibility"] - unlearn_metrics["neg_plausibility"]))
    print(f"pos_delta_rank: {pos_delta_rank}")
    print(f"neg_delta_rank: {neg_delta_rank}")
    print(f"pos_delta_p: {pos_delta_p}")
    print(f"neg_delta_p: {neg_delta_p}")

# %%
