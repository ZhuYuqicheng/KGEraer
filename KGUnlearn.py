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

class DataLoader(object):
    def __init__(self, args) -> None:
        # create dataset
        dataset_path = os.path.join(os.environ["DATA_PATH"], args.dataset)
        self.dataset = KGDataset(dataset_path, args.debug)
        args.sizes = self.dataset.get_shape()

        # load data
        self.train_examples = self.dataset.get_examples("train")
        self.valid_examples = self.dataset.get_examples("valid")
        self.test_examples = self.dataset.get_examples("test")
        self.filters = self.dataset.get_filters()

    def load_data(self, delete_triples=None):
        result = {"train_examples": self.train_examples, \
                "valid_examples": self.valid_examples, \
                "test_examples": self.test_examples, \
                "filters": self.filters
                }
        if delete_triples is not None:
            result["deleted_triples"] = delete_triples
            result["del_train_examples"] = self.dataset.get_examples("train", del_triple=delete_triples)
        return result

    def random_deletion(self, n_del):
        random_index = torch.randint(0, self.train_examples.size(0), (n_del,))
        delete_triples = self.train_examples[random_index]
        del_train_examples = self.dataset.get_examples("train", del_triple=delete_triples)
        return delete_triples, del_train_examples
    
    def degree_deletion(self, entity_rank, relation_rank, n_del):
        """entity_rank: ["high", 35], relation_rank: ["low", 2]"""
        entity_dist = pd.read_pickle("./data/WN18RR/entity_dist.pkl")
        relation_dist = pd.read_pickle("./data/WN18RR/relation_dist.pkl")

        if relation_rank[0] == "high":
            selected_pd = relation_dist.iloc[:relation_rank[1],:]
            relation_list = selected_pd["relations"].values
        elif relation_rank[0] == "low":
            selected_pd = relation_dist.iloc[-relation_rank[1]:,:]
            relation_list = selected_pd["relations"].values
        relation_mask = torch.isin(self.train_examples[:,1], torch.tensor(relation_list))
        relation_examples = self.train_examples[relation_mask]

        head_count = pd.merge(pd.DataFrame({"entities": relation_examples[:,0]}), entity_dist, how="left", on="entities")
        tail_count = pd.merge(pd.DataFrame({"entities": relation_examples[:,2]}), entity_dist, how="left", on="entities")
        entity_degree = head_count["count"].values+tail_count["count"].values
        if entity_rank[0] == "high":
            indices = np.argsort(entity_degree)[-entity_rank[1]:]
        elif entity_rank[0] == "low":
            indices = np.argsort(entity_degree)[:entity_rank[1]]
        selected_indeces = np.random.choice(indices, size=n_del, replace=False)

        delete_triples = relation_examples[selected_indeces]
        del_train_examples = self.dataset.get_examples("train", del_triple=delete_triples)

        return delete_triples, del_train_examples
        
class UnlearnPipeline(object):
    def __init__(self, args, save_dir, console, data) -> None:
        self.save_dir = save_dir
        logging.getLogger("").addHandler(console)

        self.train_examples = data["train_examples"]
        self.valid_examples = data["valid_examples"]
        self.test_examples = data["test_examples"]
        self.filters = data["filters"]
        self.deleted_triples = data["deleted_triples"]
        self.del_train_examples = data["del_train_examples"]

        # save config
        with open(os.path.join(self.save_dir, "config.json"), "w") as fjson:
            json.dump(vars(args), fjson)

    def load_model(self, args, initial_model=None):
        # create model
        if initial_model:
            model = copy.deepcopy(initial_model)
        else:
            model = getattr(models, args.model)(args)
        total = count_params(model)
        # logging.info("Total number of parameters {}".format(total))
        # device = "cuda"
        model.to(args.device)

        # get optimizer
        regularizer = getattr(regularizers, args.regularizer)(args.reg)
        optim_method = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.learning_rate)
        optimizer = KGOptimizer(model, regularizer, optim_method, args.batch_size, args.neg_sample_size,
                                bool(args.double_neg), args.device, verbose=False)
        return model, optimizer

    def get_khop_entities(self, train_examples, deleted_triples, hop):
        selected_entities = list(set(torch.cat((deleted_triples[:,0], deleted_triples[:,2])).tolist()))
        while hop > 0:
            mask  = torch.isin(train_examples[:,0], torch.Tensor(selected_entities)) | torch.isin(train_examples[:,2], torch.Tensor(selected_entities))
            hop_triples = train_examples[mask]
            add_entities = list(set(torch.cat((hop_triples[:,0], hop_triples[:,2])).tolist()))
            selected_entities += add_entities
            hop -= 1
        return selected_entities

    def train(self, args, model, optimizer, train_examples, fix_rel=False, fix_ent=False, k_hop=1):
        # get freeze_index
        if fix_ent: 
            all_index = list(range(model.entity.weight.size(0)))
            hop_entities = self.get_khop_entities(self.train_examples, self.deleted_triples, k_hop)
            freeze_index = [index for index in all_index if index not in hop_entities]
        else:
            freeze_index = []

        counter = 0
        best_mrr = None
        best_epoch = None
        # logging.info("\t Start training")
        for step in tqdm.tqdm(range(args.max_epochs)):

            # Train step
            model.train()
            train_loss = optimizer.restrict_epoch(train_examples, fix_rel=fix_rel, freeze_index=freeze_index)
            # logging.info("\t Epoch {} | average train loss: {:.4f}".format(step, train_loss))

            # Valid step
            model.eval()
            # valid_loss = optimizer.calculate_valid_loss(self.valid_examples)
            # logging.info("\t Epoch {} | average valid loss: {:.4f}".format(step, valid_loss))

            if (step + 1) % args.valid == 0:
                valid_metrics = avg_both(*model.compute_metrics(self.valid_examples, self.filters))
                # logging.info(format_metrics(valid_metrics, split="valid"))

                valid_mrr = valid_metrics["MRR"]
                if not best_mrr or valid_mrr > best_mrr:
                    best_mrr = valid_mrr
                    counter = 0
                    best_epoch = step
                    # logging.info("\t Saving model at epoch {} in {}".format(step, self.save_dir))
                    torch.save(model.cpu().state_dict(), os.path.join(self.save_dir, "buffer_model.pt"))
                    model.to(args.device)
                else:
                    counter += 1
                    if counter == args.patience:
                        # logging.info("\t Early stopping")
                        break
                    elif counter == args.patience // 2:
                        pass

        # logging.info("\t Optimization finished")
        if not best_mrr:
            torch.save(model.cpu().state_dict(), os.path.join(self.save_dir, "buffer_model.pt"))
        else:
            logging.info("\t Loading best model saved at epoch {}".format(best_epoch))
            model.load_state_dict(torch.load(os.path.join(self.save_dir, "buffer_model.pt")))

        model.to(args.device)
        model.eval()

        return model
    
    def get_hop_examples(self):
        selected_entities = list(set(torch.cat((self.deleted_triples[:,0], self.deleted_triples[:,2])).tolist()))
        mask  = torch.isin(self.del_train_examples[:,0], torch.Tensor(selected_entities)) | torch.isin(self.del_train_examples[:,2], torch.Tensor(selected_entities))
        self.hop_triples = self.del_train_examples[mask]

    def train_model(self, args, mode, initial_model=None, fix_rel=False, fix_ent=False, k_hop=1):
        start_time = time.time()
        if mode == "initial":
            model, optimizer = self.load_model(args)
            result_model = self.train(args, model, optimizer, self.train_examples)
        elif mode == "retrain":
            model, optimizer = self.load_model(args)
            result_model = self.train(args, model, optimizer, self.del_train_examples)
        elif mode == "finetune":
            model, optimizer = self.load_model(args, initial_model=initial_model)
            result_model = self.train(args, model, optimizer, self.del_train_examples, fix_rel=fix_rel, fix_ent=fix_ent, k_hop=k_hop)
        elif mode == "hop-finetune":
            model, optimizer = self.load_model(args, initial_model=initial_model)
            self.get_hop_examples()
            result_model = self.train(args, model, optimizer, self.hop_triples)
        end_time = time.time()
        logging.info(f"\t Runtime: {round(end_time-start_time,2)}s")
        return result_model

def run_influence(args, deletion_mode, n_del, degree_set=None, repeat=1, save_model=False):
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
        result_models = dict()
        result_models["initial_model"] = unlearner.train_model(args, "initial")
        if save_model: torch.save(result_models["initial_model"].cpu(), os.path.join(model_dir, "initial_model.pth"))

        result_models["retrained_model"] = unlearner.train_model(args, "retrain")
        if save_model: torch.save(result_models["retrained_model"].cpu(), os.path.join(model_dir, "retrained_model.pth"))

        result_models["finetuned_model"] = unlearner.train_model(args, "finetune", initial_model=result_models["initial_model"])
        if save_model: torch.save(result_models["finetuned_model"].cpu(), os.path.join(model_dir, "finetuned_model.pth"))

        result_models["rel_finetuned_model"] = unlearner.train_model(args, "finetune", initial_model=result_models["initial_model"], fix_rel=True)
        if save_model: torch.save(result_models["rel_finetuned_model"].cpu(), os.path.join(model_dir, "rel_finetuned_model.pth"))

        result_models["rel_ent_finetuned_model"] = unlearner.train_model(args, "finetune", initial_model=result_models["initial_model"], fix_rel=True, fix_ent=True, k_hop=1)
        if save_model: torch.save(result_models["rel_ent_finetuned_model"].cpu(), os.path.join(model_dir, "rel_ent_finetuned_model.pth"))

    # return result_models, result_runtimes, evaluator

def model_test(args):
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

    data_loader = DataLoader(args)
    data = data_loader.load_data()

    counter = 0
    best_mrr = None
    best_epoch = None
    # logging.info("\t Start training")
    for step in tqdm.tqdm(range(args.max_epochs)):
        model = getattr(models, args.model)(args)
        model.to(args.device)
        # get optimizer
        regularizer = getattr(regularizers, args.regularizer)(args.reg)
        optim_method = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.learning_rate)
        optimizer = KGOptimizer(model, regularizer, optim_method, args.batch_size, args.neg_sample_size,
                                bool(args.double_neg), args.device, verbose=False)
        
        # Train step
        model.train()
        train_loss = optimizer.restrict_epoch(data["train_examples"])
        # logging.info("\t Epoch {} | average train loss: {:.4f}".format(step, train_loss))

        # Valid step
        model.eval()
        # valid_loss = optimizer.calculate_valid_loss(self.valid_examples)
        # logging.info("\t Epoch {} | average valid loss: {:.4f}".format(step, valid_loss))

        if (step + 1) % args.valid == 0:
            valid_metrics = avg_both(*model.compute_metrics(data["valid_examples"], data["filters"]))
            # logging.info(format_metrics(valid_metrics, split="valid"))

            valid_mrr = valid_metrics["MRR"]
            if not best_mrr or valid_mrr > best_mrr:
                best_mrr = valid_mrr
                counter = 0
                best_epoch = step
                # logging.info("\t Saving model at epoch {} in {}".format(step, self.save_dir))
                torch.save(model.cpu().state_dict(), os.path.join(save_dir, "buffer_model.pt"))
                model.to(args.device)
            else:
                counter += 1
                if counter == args.patience:
                    # logging.info("\t Early stopping")
                    break
                elif counter == args.patience // 2:
                    pass

    # logging.info("\t Optimization finished")
    if not best_mrr:
        torch.save(model.cpu().state_dict(), os.path.join(save_dir, "buffer_model.pt"))
    else:
        logging.info("\t Loading best model saved at epoch {}".format(best_epoch))
        model.load_state_dict(torch.load(os.path.join(save_dir, "buffer_model.pt")))

    model.to(args.device)
    model.eval()

    valid_metrics = avg_both(*model.compute_metrics(data["valid_examples"], data["filters"]))
    logging.info(format_metrics(valid_metrics, split="valid"))
    test_metrics = avg_both(*model.compute_metrics(data["test_examples"], data["filters"]))
    logging.info(format_metrics(test_metrics, split="test"))
    return model