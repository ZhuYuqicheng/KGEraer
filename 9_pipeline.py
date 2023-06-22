#%%
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
    def __init__(self, args, console) -> None:
        logging.getLogger("").addHandler(console)
        # create dataset
        dataset_path = os.path.join(os.environ["DATA_PATH"], args.dataset)
        self.dataset = KGDataset(dataset_path, args.debug)
        args.sizes = self.dataset.get_shape()

        # load data
        logging.info("\t " + str(self.dataset.get_shape()))
        self.train_examples = self.dataset.get_examples("train")
        self.valid_examples = self.dataset.get_examples("valid")
        self.test_examples = self.dataset.get_examples("test")
        self.filters = self.dataset.get_filters()

    def load_data(self):
        return {"train_examples": self.train_examples, \
                "valid_examples": self.valid_examples, \
                "test_examples": self.test_examples, \
                "filters": self.filters
        }

    def random_deletion(self, n_del):
        random_index = torch.randint(0, self.train_examples.size(0), (n_del,))
        delete_triples = self.train_examples[random_index]
        del_train_examples = self.dataset.get_examples("train", del_triple=delete_triples)
        return delete_triples, del_train_examples
    
    def degree_deletion(self):
        pass
        
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

        # set random seed
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)

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

    def train_model(self, args, mode, initial_model=None, fix_rel=False, fix_ent=False, k_hop=1, rep=1):
        result_models, runtimes = [], []
        for _ in range(rep):
            start_time = time.time()
            if mode == "initial":
                model, optimizer = self.load_model(args)
                result_model = self.train(args, model, optimizer, self.train_examples)
            elif mode == "retrain":
                model, optimizer = self.load_model(args)
                result_model = self.train(args, model, optimizer, self.del_train_examples)
            elif mode == "finetune":
                model, optimizer = self.load_model(args, initial_model=initial_model)
                result_model = self.train(args, model, optimizer, self.del_train_examples, fix_rel=False, fix_ent=False, k_hop=1)
            elif mode == "hop-finetune":
                model, optimizer = self.load_model(args, initial_model=initial_model)
                self.get_hop_examples()
                result_model = self.train(args, model, optimizer, self.hop_triples)
            end_time = time.time()
            result_models.append(result_model)
            runtimes.append(end_time-start_time)
            mean_runtime = sum(runtimes)/len(runtimes)
        return result_models, mean_runtime

    def run(self, args):
        initial_model, _ = self.train_model(args, "initial")

        result_models, result_runtimes = dict(), dict()
        result_models["retrained_model"], result_runtimes["retrained_time"] = self.train_model(args, "retrain", rep=5)
        result_models["finetuned_model"], result_runtimes["finetuned_time"] = self.train_model(args, "finetune", initial_model=initial_model[0], rep=5)
        result_models["rel_finetuned_model"], result_runtimes["rel_finetuned_time"] = self.train_model(args, "finetune", initial_model=initial_model[0], fix_rel=True, rep=5)
        #result_models["ent_finetuned_model"], result_runtimes["ent_finetuned_time"] = self.train_model(args, "finetune", initial_model=initial_model, fix_ent=True, k_hop=1)
        #result_models["rel_ent_finetuned_model"], result_runtimes["rel_ent_finetuned_time"] = self.train_model(args, "finetune", initial_model=initial_model, fix_rel=True, fix_ent=True, k_hop=1)
        result_models["hop_finetuned_model"], result_runtimes["hop_finetuned_model"] = self.train_model(args, "hop-finetune", initial_model=initial_model[0], rep=5)
        return result_models, result_runtimes

class Evaluator(object):
    def __init__(self, data) -> None:
        self.train_examples = data["train_examples"]
        self.valid_examples = data["valid_examples"]
        self.test_examples = data["test_examples"]
        self.filters = data["filters"]
        self.deleted_triples = data["deleted_triples"]
        self.del_train_examples = data["del_train_examples"]

    def model_utility(self, models):
        valid_MR, valid_MRR, valid_hits = [],[],[]
        test_MR, test_MRR, test_hits = [],[],[]
        for model in models:
            # Validation metrics
            valid_metric = avg_both(*model.compute_metrics(self.valid_examples, self.filters))
            valid_MR.append(valid_metric["MR"])
            valid_MRR.append(valid_metric["MRR"])
            valid_hits.append(valid_metric["hits@[1,3,10]"])
            # Test metrics
            test_metric = avg_both(*model.compute_metrics(self.test_examples, self.filters))
            test_MR.append(test_metric["MR"])
            test_MRR.append(test_metric["MRR"])
            test_hits.append(test_metric["hits@[1,3,10]"])
        
        valid_metrics = {"MR": sum(valid_MR)/len(valid_MR), \
                         "MRR": sum(valid_MRR)/len(valid_MRR), \
                         "hits@[1,3,10]": torch.mean(torch.stack(valid_hits), dim=0)}
        test_metrics = {"MR": sum(test_MR)/len(test_MR), \
                         "MRR": sum(test_MRR)/len(test_MRR), \
                         "hits@[1,3,10]": torch.mean(torch.stack(test_hits), dim=0)}
        return valid_metrics, test_metrics

    def rank_evaluation(self, models, batch_size=1000):
        pos_ranks_list, neg_ranks_list = [], []
        for model in models:
            pos_ranks = model.get_ranking(self.del_train_examples, {**self.filters["lhs"], **self.filters["rhs"]}, batch_size=batch_size)
            neg_ranks = model.get_ranking(self.deleted_triples, {**self.filters["lhs"], **self.filters["rhs"]}, batch_size=batch_size)
            pos_ranks_list.append(pos_ranks)
            neg_ranks_list.append(neg_ranks)
        average_pos_ranks = sum(pos_ranks_list)/len(pos_ranks_list)
        average_neg_ranks = sum(neg_ranks_list)/len(neg_ranks_list)
        return average_pos_ranks, average_neg_ranks
    
    def evaluate(self, models, runtimes):
        result_models = list(models.values())
        retrained_model = result_models[0]
        baseline_models = result_models[1:]

        model_names = list(models.keys())
        model_names = model_names[1:]
        
        runtimes = list(runtimes.values())
        retrained_runtime = runtimes[0]
        baseline_runtimes = runtimes[1:]

        retrain_valid_metrics, retrain_test_metrics = self.model_utility(retrained_model)
        retrain_pos_ranks, retrain_neg_ranks = self.rank_evaluation(retrained_model)
        logging.info(f"For retrained model:")
        logging.info(format_metrics(retrain_valid_metrics, split="valid"))
        logging.info(format_metrics(retrain_test_metrics, split="test"))
        logging.info("--------------------------------------------------")

        for ind, baseline_model in enumerate(baseline_models):
            # evaluate utility
            valid_metrics, test_metrics = self.model_utility(baseline_model)
            delta_valid_metrics = dict()
            delta_valid_metrics["MR"] = abs(retrain_valid_metrics["MR"]-valid_metrics["MR"])
            delta_valid_metrics["MRR"] = abs(retrain_valid_metrics["MRR"]-valid_metrics["MRR"])
            delta_valid_metrics["hits@[1,3,10]"] = torch.abs(retrain_valid_metrics["hits@[1,3,10]"]-valid_metrics["hits@[1,3,10]"])

            delta_test_metrics = dict()
            delta_test_metrics["MR"] = abs(retrain_test_metrics["MR"]-test_metrics["MR"])
            delta_test_metrics["MRR"] = abs(retrain_test_metrics["MRR"]-test_metrics["MRR"])
            delta_test_metrics["hits@[1,3,10]"] = torch.abs(retrain_test_metrics["hits@[1,3,10]"]-test_metrics["hits@[1,3,10]"])

            logging.info(f"For baseline {model_names[ind]}:")
            logging.info(f"\t Model Utility:")
            logging.info(format_metrics(valid_metrics, split="valid"))
            logging.info(format_metrics(test_metrics, split="test"))
            logging.info(f"\t Delta Utility:")
            logging.info(format_metrics(delta_valid_metrics, split="delta valid"))
            logging.info(format_metrics(delta_test_metrics, split="delta test"))

            # evaluate embeddings
            average_l2_distance = []
            std_l2_distance = []
            max_l2_distance = []
            min_l2_distance = []
            for single_retrain_model, single_baseline_model in zip(retrained_model, baseline_model):
                l2_distance = torch.sum(torch.square(single_retrain_model.entity.weight.data - single_baseline_model.entity.weight.data), dim=1)
                average_l2_distance.append(torch.mean(l2_distance).item())
                std_l2_distance.append(torch.std(l2_distance).item())
                max_l2_distance.append(torch.max(l2_distance).item())
                min_l2_distance.append(torch.min(l2_distance).item())
            average_l2_distance = sum(average_l2_distance)/len(average_l2_distance)
            std_l2_distance = sum(std_l2_distance)/len(std_l2_distance)
            max_l2_distance = sum(max_l2_distance)/len(max_l2_distance)
            min_l2_distance = sum(min_l2_distance)/len(min_l2_distance)
            logging.info(f"\t Embedding distance (l2):")
            logging.info(f"\t Mean(std): {round(average_l2_distance,2)}({round(std_l2_distance,2)}); min/max:{round(min_l2_distance,2)}/{round(max_l2_distance,2)}")

            # evaluate ranks
            pos_ranks, neg_ranks = self.rank_evaluation(baseline_model)
            baseline_pos_ranks = torch.mean(torch.abs(retrain_pos_ranks-pos_ranks)).item()
            baseline_neg_ranks = torch.mean(torch.abs(retrain_neg_ranks-neg_ranks)).item()
            logging.info(f"\t Average Delta Ranks: ")
            logging.info(f"\t Positve: {round(baseline_pos_ranks,1)}; Negative: {round(baseline_neg_ranks,1)}")

            # evaluate runtime
            delta_runtime = retrained_runtime - baseline_runtimes[ind]
            logging.info(f"\t Delta Runtime: {round(delta_runtime,2)}")
    
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
    # load data
    data_loader = DataLoader(args, console)
    data = data_loader.load_data()
    data["deleted_triples"], data["del_train_examples"] = data_loader.random_deletion(n_del=10)
    # run unlearning training
    unlearner = UnlearnPipeline(args, save_dir, console, data)
    unlearn_models, unlearn_runtimes = unlearner.run(args)
    # evaluate unlearning
    evaluator = Evaluator(data)
    evaluator.evaluate(unlearn_models, unlearn_runtimes)

# %%
