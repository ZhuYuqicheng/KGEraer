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
from matplotlib import pyplot as plt 

import torch
import torch.optim
from torch import nn

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
    
    def get_hop_triples(self, data, n_hop):
        deleted_triples = data["deleted_triples"]
        del_train_examples = data["del_train_examples"]

        hop_triples = copy.deepcopy(deleted_triples)
        for _ in range(n_hop):
            selected_entities = list(set(torch.cat((hop_triples[:,0], hop_triples[:,2])).tolist()))
            mask  = torch.isin(del_train_examples[:,0], torch.Tensor(selected_entities)) | torch.isin(del_train_examples[:,2], torch.Tensor(selected_entities))
            hop_triples = del_train_examples[mask]
        coverage = round(hop_triples.size(0)/del_train_examples.size(0)*100,2)
        print(f"{hop_triples.size(0)}/{del_train_examples.size(0)} -> Cover: {coverage}%")
        return hop_triples, coverage

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
    
class Evaluator(object):
    def __init__(self, data) -> None:
        self.train_examples = data["train_examples"]
        self.valid_examples = data["valid_examples"]
        self.test_examples = data["test_examples"]
        self.filters = data["filters"]
        self.deleted_triples = data["deleted_triples"]
        self.del_train_examples = data["del_train_examples"]

    def model_utility(self, model):
        model.to(model.device)
        valid_metric = avg_both(*model.compute_metrics(self.valid_examples, self.filters))
        test_metric = avg_both(*model.compute_metrics(self.test_examples, self.filters))
        return valid_metric, test_metric
    
    def rank_evaluation(self, model1, model2, hop_triples=None, batch_size=1000):
        model1.to(model1.device)
        model2.to(model2.device)
        if hop_triples is None:
            pos_ranks1 = model1.get_ranking(self.del_train_examples, {**self.filters["lhs"], **self.filters["rhs"]}, batch_size=batch_size)
            pos_ranks2 = model2.get_ranking(self.del_train_examples, {**self.filters["lhs"], **self.filters["rhs"]}, batch_size=batch_size)
        else:
            pos_ranks1 = model1.get_ranking(hop_triples, {**self.filters["lhs"], **self.filters["rhs"]}, batch_size=batch_size)
            pos_ranks2 = model2.get_ranking(hop_triples, {**self.filters["lhs"], **self.filters["rhs"]}, batch_size=batch_size)
        delta_pos_rank = torch.mean(torch.abs(pos_ranks1 - pos_ranks2))

        neg_ranks1 = model1.get_ranking(self.deleted_triples, {**self.filters["lhs"], **self.filters["rhs"]}, batch_size=batch_size)
        neg_ranks2 = model2.get_ranking(self.deleted_triples, {**self.filters["lhs"], **self.filters["rhs"]}, batch_size=batch_size)
        delta_neg_rank = torch.mean(torch.abs(neg_ranks1 - neg_ranks2))
        neg_rank_before, neg_rank_after = torch.mean(neg_ranks1), torch.mean(neg_ranks2)
        return delta_pos_rank, delta_neg_rank, neg_rank_before, neg_rank_after
    
    def get_prob(self, model, training_examples, deleted_triples, batch_size=1000):
        sig = nn.Sigmoid()
        b_begin = 0
        pos_plausibility = torch.empty(0).to(model.device)
        with torch.no_grad():
            # get positive plausibilities
            while b_begin < training_examples.shape[0]:
                input_batch = training_examples[
                              b_begin:b_begin + batch_size
                              ].to(model.device)
                b_begin += batch_size
                positive_score, _ = model.forward(input_batch)
                pos_plausibility = torch.cat((pos_plausibility, sig(positive_score)), 0)
            # get negative plausibilities
            negative_score, _ = model.forward(deleted_triples.to(model.device))
            neg_plausibility = sig(negative_score)
        return pos_plausibility, neg_plausibility

    def prob_evaluation(self, model1, model2, hop_triples=None):
        model1.to(model1.device)
        model2.to(model2.device)
        deleted_triples = self.deleted_triples
        if hop_triples is None:
            training_examples = self.del_train_examples
        else:
            training_examples = hop_triples
        pos_p1, neg_p1 = self.get_prob(model1, training_examples, deleted_triples)
        pos_p2, neg_p2 = self.get_prob(model2, training_examples, deleted_triples)
        delta_pos_p = torch.mean(torch.abs(pos_p1 - pos_p2))
        delta_neg_p = torch.mean(torch.abs(neg_p1 - neg_p2))
        neg_p_before, neg_p_after = torch.mean(neg_p1), torch.mean(neg_p2)
        return delta_pos_p, delta_neg_p, neg_p_before, neg_p_after

    # def rank_evaluation(self, models, batch_size=1000):
    #     pos_ranks_list, neg_ranks_list = [], []
    #     for model in models:
    #         pos_ranks = model.get_ranking(self.del_train_examples, {**self.filters["lhs"], **self.filters["rhs"]}, batch_size=batch_size)
    #         neg_ranks = model.get_ranking(self.deleted_triples, {**self.filters["lhs"], **self.filters["rhs"]}, batch_size=batch_size)
    #         pos_ranks_list.append(pos_ranks)
    #         neg_ranks_list.append(neg_ranks)
    #     average_pos_ranks = sum(pos_ranks_list)/len(pos_ranks_list)
    #     average_neg_ranks = sum(neg_ranks_list)/len(neg_ranks_list)
    #     return average_pos_ranks, average_neg_ranks
    
    def visual_delta_embedding(self, axs, model1, model2):
        delta_entity_embedding = torch.sum(torch.square(model1.entity.weight.data - model2.entity.weight.data), dim=1).cpu()
        delta_rel_embedding = torch.sum(torch.square(model1.rel.weight.data - model2.rel.weight.data), dim=1).cpu()

        highlight_x = list(set(torch.cat((self.deleted_triples[:,0], self.deleted_triples[:,2])).tolist()))
        highlight_y = [delta_entity_embedding[ind].item() for ind in highlight_x]
        y = pd.Series(delta_entity_embedding).sort_values(ascending=False).values
        x = range(len(y))
        highlight_x = [np.where(y==m)[0] for m in highlight_y]

        # fig, axs = plt.subplots(2)
        axs[0].plot(x, y)
        axs[0].plot(highlight_x, highlight_y, "o")

        highlight_x = list(set(self.deleted_triples[:,1].tolist()))
        highlight_y = [delta_rel_embedding[ind].item() for ind in highlight_x]
        y = pd.Series(delta_rel_embedding).sort_values(ascending=False).values
        x = range(len(y))
        highlight_x = [np.where(y==m)[0] for m in highlight_y]

        axs[1].plot(x, y)
        axs[1].plot(highlight_x, highlight_y, "o")
    
    # def evaluate(self, models, runtimes):
    #     result_models = list(models.values())
    #     retrained_model = result_models[0]
    #     baseline_models = result_models[1:]

    #     model_names = list(models.keys())
    #     model_names = model_names[1:]
        
    #     runtimes = list(runtimes.values())
    #     retrained_runtime = runtimes[0]
    #     baseline_runtimes = runtimes[1:]

    #     retrain_valid_metrics, retrain_test_metrics = self.model_utility(retrained_model)
    #     retrain_pos_ranks, retrain_neg_ranks = self.rank_evaluation(retrained_model)
    #     logging.info(f"For retrained model:")
    #     logging.info(format_metrics(retrain_valid_metrics, split="valid"))
    #     logging.info(format_metrics(retrain_test_metrics, split="test"))
    #     logging.info("--------------------------------------------------")

    #     for ind, baseline_model in enumerate(baseline_models):
    #         # evaluate utility
    #         valid_metrics, test_metrics = self.model_utility(baseline_model)
    #         delta_valid_metrics = dict()
    #         delta_valid_metrics["MR"] = abs(retrain_valid_metrics["MR"]-valid_metrics["MR"])
    #         delta_valid_metrics["MRR"] = abs(retrain_valid_metrics["MRR"]-valid_metrics["MRR"])
    #         delta_valid_metrics["hits@[1,3,10]"] = torch.abs(retrain_valid_metrics["hits@[1,3,10]"]-valid_metrics["hits@[1,3,10]"])

    #         delta_test_metrics = dict()
    #         delta_test_metrics["MR"] = abs(retrain_test_metrics["MR"]-test_metrics["MR"])
    #         delta_test_metrics["MRR"] = abs(retrain_test_metrics["MRR"]-test_metrics["MRR"])
    #         delta_test_metrics["hits@[1,3,10]"] = torch.abs(retrain_test_metrics["hits@[1,3,10]"]-test_metrics["hits@[1,3,10]"])

    #         logging.info(f"For baseline {model_names[ind]}:")
    #         logging.info(f"\t Model Utility:")
    #         logging.info(format_metrics(valid_metrics, split="valid"))
    #         logging.info(format_metrics(test_metrics, split="test"))
    #         logging.info(f"\t Delta Utility:")
    #         logging.info(format_metrics(delta_valid_metrics, split="delta valid"))
    #         logging.info(format_metrics(delta_test_metrics, split="delta test"))

    #         # evaluate embeddings
    #         average_l2_distance = []
    #         std_l2_distance = []
    #         max_l2_distance = []
    #         min_l2_distance = []
    #         for single_retrain_model, single_baseline_model in zip(retrained_model, baseline_model):
    #             l2_distance = torch.sum(torch.square(single_retrain_model.entity.weight.data - single_baseline_model.entity.weight.data), dim=1)
    #             average_l2_distance.append(torch.mean(l2_distance).item())
    #             std_l2_distance.append(torch.std(l2_distance).item())
    #             max_l2_distance.append(torch.max(l2_distance).item())
    #             min_l2_distance.append(torch.min(l2_distance).item())
    #         average_l2_distance = sum(average_l2_distance)/len(average_l2_distance)
    #         std_l2_distance = sum(std_l2_distance)/len(std_l2_distance)
    #         max_l2_distance = sum(max_l2_distance)/len(max_l2_distance)
    #         min_l2_distance = sum(min_l2_distance)/len(min_l2_distance)
    #         logging.info(f"\t Embedding distance (l2):")
    #         logging.info(f"\t Mean(std): {round(average_l2_distance,2)}({round(std_l2_distance,2)}); min/max:{round(min_l2_distance,2)}/{round(max_l2_distance,2)}")

    #         # evaluate ranks
    #         pos_ranks, neg_ranks = self.rank_evaluation(baseline_model)
    #         baseline_pos_ranks = torch.mean(torch.abs(retrain_pos_ranks-pos_ranks)).item()
    #         baseline_neg_ranks = torch.mean(torch.abs(retrain_neg_ranks-neg_ranks)).item()
    #         logging.info(f"\t Average Delta Ranks: ")
    #         logging.info(f"\t Positve: {round(baseline_pos_ranks,1)}; Negative: {round(baseline_neg_ranks,1)}")

    #         # evaluate runtime
    #         delta_runtime = retrained_runtime - baseline_runtimes[ind]
    #         logging.info(f"\t Delta Runtime: {round(delta_runtime,2)}")

def evaluate_utility(args, path, model_case):
    data_loader = DataLoader(args)
    valid_MR, valid_MRR, valid_hits = [], [], []
    test_MR, test_MRR, test_hits = [], [], []
    for i in range(5):
        data = data_loader.load_data(delete_triples=torch.load(path+str(i+1)+"/deleted_triples.pt"))
        evaluator = Evaluator(data)
        model = torch.load(path+str(i+1)+"/"+model_case+".pth")
        valid_metric, test_metric = evaluator.model_utility(model)
        valid_MR.append(valid_metric["MR"])
        valid_MRR.append(valid_metric["MRR"])
        valid_hits.append(valid_metric["hits@[1,3,10]"])
        test_MR.append(test_metric["MR"])
        test_MRR.append(test_metric["MRR"])
        test_hits.append(test_metric["hits@[1,3,10]"])

    valid_metrics = {"MR": sum(valid_MR)/len(valid_MR), \
                         "MRR": sum(valid_MRR)/len(valid_MRR), \
                         "hits@[1,3,10]": torch.mean(torch.stack(valid_hits), dim=0)}
    test_metrics = {"MR": sum(test_MR)/len(test_MR), \
                        "MRR": sum(test_MRR)/len(test_MRR), \
                        "hits@[1,3,10]": torch.mean(torch.stack(test_hits), dim=0)}
    print(format_metrics(valid_metrics, split="valid"))
    print(format_metrics(test_metrics, split="test"))

# load data
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
        random_seed = 0
    )

    # 5 times 1-degree deletion
    # path = "/workspace/LLKGE/KGEmb/logs/06_29/WN18RR/TransE_13_08_34/1_degree_1/" # low low
    # path = "/workspace/LLKGE/KGEmb/logs/06_29/WN18RR/TransE_13_14_34/1_degree_1/" # high high
    # path = "/workspace/LLKGE/KGEmb/logs/06_29/WN18RR/TransE_13_10_56/1_degree_1/" # high low
    # path = "/workspace/LLKGE/KGEmb/logs/06_29/WN18RR/TransE_13_13_39/1_degree_1/" # low high
    # path = "/workspace/LLKGE/KGEmb/logs/06_30/WN18RR/TransE_12_25_45/1_degree_1/" # highest triple
    # path = "/workspace/LLKGE/KGEmb/logs/06_30/WN18RR/TransE_12_26_44/1_degree_1/" # lowest triple
    # initial_model, retrained_model, finetuned_model, rel_finetuned_model, rel_ent_finetuned_model

    # ---------------------------- delta embedding visualization ---------------------------- #
    # path = "/workspace/LLKGE/KGEmb/logs/06_30/WN18RR/TransE_12_26_44/1_degree_"
    # data_loader = DataLoader(args)
    # fig, axs = plt.subplots(2)
    # for i in range(5):
    #     data = data_loader.load_data(delete_triples=torch.load(path+str(i+1)+"/deleted_triples.pt"))
    #     evaluator = Evaluator(data)
    #     model1 = torch.load(path+str(i+1)+"/initial_model.pth")
    #     model2 = torch.load(path+str(i+1)+"/finetuned_model.pth")
    #     evaluator.visual_delta_embedding(axs, model1, model2)
    # plt.show()

    # ---------------------------- model utility evaluation ---------------------------- #
    # path = "/workspace/LLKGE/KGEmb/logs/06_29/WN18RR/TransE_13_13_39/1_degree_"
    # print("Model Utility: initial_model")
    # evaluate_utility(args, path, "initial_model")
    # print("Model Utility: retrained_model")
    # evaluate_utility(args, path, "retrained_model")
    # print("Model Utility: finetuned_model")
    # evaluate_utility(args, path, "finetuned_model")
    # print("Model Utility: rel_finetuned_model")
    # evaluate_utility(args, path, "rel_finetuned_model")
    # print("Model Utility: rel_ent_finetuned_model")
    # evaluate_utility(args, path, "rel_ent_finetuned_model")

    # ---------------------------- delta output evaluation ---------------------------- #
    # for case in ["TransE_12_25_45", "TransE_12_26_44"]:
    #     path = "/workspace/LLKGE/KGEmb/logs/06_30/WN18RR/"+case+"/1_degree_"
    #     data_loader = DataLoader(args)
    #     for model_path in ["/retrained_model.pth", "/finetuned_model.pth", "/rel_finetuned_model.pth", "/rel_ent_finetuned_model.pth"]:
    #         print(f"Case {case}, init vs {model_path}..")
    #         delta_pos_ranks, delta_neg_ranks = [], []
    #         for i in range(5):
    #             data = data_loader.load_data(delete_triples=torch.load(path+str(i+1)+"/deleted_triples.pt"))
    #             evaluator = Evaluator(data)
    #             model1 = torch.load(path+str(i+1)+"/initial_model.pth")
    #             model2 = torch.load(path+str(i+1)+model_path)
    #             delta_pos_rank, delta_neg_rank = evaluator.rank_evaluation(model1, model2)
    #             delta_pos_ranks.append(delta_pos_rank)
    #             delta_neg_ranks.append(delta_neg_rank)
    #             print(f"\t delta postive rank: {delta_pos_rank}, delta negative rank: {delta_neg_rank}")
    #         print(f"\t average delta postive rank: {sum(delta_pos_ranks)/len(delta_pos_ranks)}, delta negative rank: {sum(delta_neg_ranks)/len(delta_neg_ranks)}")

    # for case in ["TransE_12_25_45", "TransE_12_26_44"]:
    #     path = "/workspace/LLKGE/KGEmb/logs/06_30/WN18RR/"+case+"/1_degree_"
    #     data_loader = DataLoader(args)
    #     for model_path in ["/retrained_model.pth", "/finetuned_model.pth", "/rel_finetuned_model.pth", "/rel_ent_finetuned_model.pth"]:
    #         print(f"Case {case}, init vs {model_path}..")
    #         delta_pos_ps, delta_neg_ps = [], []
    #         neg_p_befores, neg_p_afters = [], []
    #         for i in range(5):
    #             data = data_loader.load_data(delete_triples=torch.load(path+str(i+1)+"/deleted_triples.pt"))
    #             evaluator = Evaluator(data)
    #             model1 = torch.load(path+str(i+1)+"/initial_model.pth")
    #             model2 = torch.load(path+str(i+1)+model_path)
    #             delta_pos_p, delta_neg_p, neg_p_before, neg_p_after = evaluator.prob_evaluation(model1, model2)
    #             delta_pos_ps.append(delta_pos_p)
    #             delta_neg_ps.append(delta_neg_p)
    #             neg_p_befores.append(neg_p_before)
    #             neg_p_afters.append(neg_p_after)
    #             print(f"\t delta postive rank: {delta_pos_p}, delta negative rank: {delta_neg_p}")
    #         print("-----------------------------")
    #         print(f"\t average delta postive rank: {sum(delta_pos_ps)/len(delta_pos_ps)}")
    #         print(f"delta negative rank: {sum(delta_neg_ps)/len(delta_neg_ps)} ({sum(neg_p_befores)/len(neg_p_befores)} -> {sum(neg_p_afters)/len(neg_p_afters)})")

    # ---------------------------- hop_triple evaluation ---------------------------- #
    # for case in ["TransE_13_14_34"]:
    #     path = "/workspace/LLKGE/KGEmb/logs/06_29/WN18RR/"+case+"/1_degree_"
    #     data_loader = DataLoader(args)
    #     fig, ax1 = plt.subplots()
    #     for i in range(5):
    #         hop_ranks = []
    #         coverages = []
    #         data = data_loader.load_data(delete_triples=torch.load(path+str(i+1)+"/deleted_triples.pt"))
    #         for n_hop in range(10):
    #             print(f"---------------- generate {n_hop}-hop triples ----------------")
    #             hop_triples, coverage = data_loader.get_hop_triples(data, n_hop+1)
    #             evaluator = Evaluator(data)
    #             model1 = torch.load(path+str(i+1)+"/initial_model.pth")
    #             model2 = torch.load(path+str(i+1)+"/finetuned_model.pth")
    #             delta_pos_rank, delta_neg_rank, neg_rank_before, neg_rank_after = evaluator.rank_evaluation(model1, model2, hop_triples=hop_triples)
    #             hop_ranks.append(delta_pos_rank)
    #             coverages.append(coverage)
    #             print(f"delta postive rank: {delta_pos_rank}, delta negative rank: {delta_neg_rank} ({neg_rank_before} -> {neg_rank_after})")
    #         print(f"---------------- generate all triples ----------------")
    #         delta_pos_rank, delta_neg_rank, neg_rank_before, neg_rank_after = evaluator.rank_evaluation(model1, model2)
    #         print(f"delta postive rank: {delta_pos_rank}, delta negative rank: {delta_neg_rank} ({neg_rank_before} -> {neg_rank_after})")
    #         hop_ranks.append(delta_pos_rank)
    #         coverages.append(100)
    #         ax1.plot(coverages, hop_ranks)
    # plt.show()

    # for case in ["TransE_13_14_34"]:
    #     path = "/workspace/LLKGE/KGEmb/logs/06_29/WN18RR/"+case+"/1_degree_"
    #     data_loader = DataLoader(args)
    #     fig, ax1 = plt.subplots()
    #     for i in range(5):
    #         hop_ps = []
    #         coverages = []
    #         data = data_loader.load_data(delete_triples=torch.load(path+str(i+1)+"/deleted_triples.pt"))
    #         for n_hop in range(10):
    #             print(f"---------------- generate {n_hop}-hop triples ----------------")
    #             hop_triples, coverage = data_loader.get_hop_triples(data, n_hop+1)
    #             evaluator = Evaluator(data)
    #             model1 = torch.load(path+str(i+1)+"/initial_model.pth")
    #             model2 = torch.load(path+str(i+1)+"/finetuned_model.pth")
    #             delta_pos_p, delta_neg_p, neg_p_before, neg_p_after = evaluator.prob_evaluation(model1, model2, hop_triples=hop_triples)
    #             hop_ps.append(delta_pos_p.item())
    #             coverages.append(coverage)
    #             print(f"delta postive p: {delta_pos_p}, delta negative p: {delta_neg_p} ({neg_p_before} -> {neg_p_after})")
    #         print(f"---------------- generate all triples ----------------")
    #         delta_pos_p, delta_neg_p, neg_p_before, neg_p_after = evaluator.prob_evaluation(model1, model2)
    #         print(f"delta postive p: {delta_pos_p}, delta negative p: {delta_neg_p} ({neg_p_before} -> {neg_p_after})")
    #         hop_ps.append(delta_pos_p.item())
    #         coverages.append(100)
    #         ax1.plot(coverages, hop_ps)
    # plt.show()

# %%
