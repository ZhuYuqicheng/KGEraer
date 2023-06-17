
import torch
import pandas as pd

def sample_entity(entity_dist, mode):
    entity_dist = entity_dist[entity_dist["count"]!=0]
    if mode == "high":
        selected_pd = entity_dist.iloc[:35,:]
        entity_list = selected_pd["entities"].values
    elif mode == "low":
        selected_pd = entity_dist.iloc[-35:,:]
        entity_list = selected_pd["entities"].values
    return torch.tensor(entity_list)

def sample_relation(relation_dist, mode):
    if mode == "high":
        selected_pd = relation_dist.iloc[:2,:]
        relation_list = selected_pd["relations"].values
    elif mode == "low":
        selected_pd = relation_dist.iloc[-2:,:]
        relation_list = selected_pd["relations"].values
    return torch.tensor(relation_list)

def special_deletion(args, train_data):
    entity_dist = pd.read_pickle("./data/WN18RR/entity_dist.pkl")
    relation_dist = pd.read_pickle("./data/WN18RR/relation_dist.pkl")

    train_data = train_data[train_data[:,1]<args.sizes[1]//2]

    delete_triples = dict()
    for head_mode in ["high", "low"]:
        for rel_mode in ["high", "low"]:
            for tail_mode in ["high", "low"]:
                head_mask = torch.isin(train_data[:,0], sample_entity(entity_dist, head_mode))
                relation_mask = torch.isin(train_data[:,1], sample_relation(relation_dist, rel_mode))
                tail_mask = torch.isin(train_data[:,2], sample_entity(entity_dist, tail_mode))
                select_data = train_data[head_mask|relation_mask|tail_mask]
                random_index = torch.randint(0, select_data.size(0), (1,)).item()
                delete_triples[(head_mode,rel_mode,tail_mode)] = select_data[random_index]
    return delete_triples

def random_deletion(train_data, n_del):
    random_index = torch.randint(0, train_data.size(0), (n_del,))
    delete_triples = train_data[random_index]
    return delete_triples
