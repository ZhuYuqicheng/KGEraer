#%%
import collections
import os
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

def to_np_array(dataset_file, ent2idx, rel2idx):
    examples = []
    with open(dataset_file, "r") as lines:
        for line in lines:
            lhs, rel, rhs = line.strip().split("\t")
            try:
                examples.append([ent2idx[lhs], rel2idx[rel], ent2idx[rhs]])
            except ValueError:
                continue
    return np.array(examples).astype("int64")

def get_filters(examples, n_relations):
    lhs_filters = collections.defaultdict(set)
    rhs_filters = collections.defaultdict(set)
    for lhs, rel, rhs in examples:
        rhs_filters[(lhs, rel)].add(rhs)
        lhs_filters[(rhs, rel + n_relations)].add(lhs)
    lhs_final = {}
    rhs_final = {}
    for k, v in lhs_filters.items():
        lhs_final[k] = sorted(list(v))
    for k, v in rhs_filters.items():
        rhs_final[k] = sorted(list(v))
    return lhs_final, rhs_final

def process_dataset(path):
    ent2idx, rel2idx = get_index(dataset_path)
    examples = {}
    splits = ["train", "valid", "test"]
    for split in splits:
        dataset_file = os.path.join(path, split+".txt")
        examples[split] = to_np_array(dataset_file, ent2idx, rel2idx)
    all_examples = np.concatenate([examples[split] for split in splits], axis=0)
    lhs_skip, rhs_skip = get_filters(all_examples, len(rel2idx))
    filters = {"lhs": lhs_skip, "rhs": rhs_skip}
    return examples, filters

def get_index(path):
    entity2id_path = os.path.join(path, "entity2id.txt")
    rel2id_path = os.path.join(path, "relation2id.txt")

    ent2idx, rel2idx = {}, {}
    with open(entity2id_path, 'r') as file:
        for line in file:
            entity, entity_id = line.strip().split('\t')
            ent2idx[entity] = entity_id
    with open(rel2id_path, 'r') as file:
        for line in file:
            relation, relation_id = line.strip().split('\t')
            rel2idx[relation] = relation_id
    return ent2idx, rel2idx

def dataset_preprocessing(dataset_path):
    dataset_examples, dataset_filters = process_dataset(dataset_path)
    for dataset_split in ["train", "valid", "test"]:
        save_path = os.path.join(dataset_path, dataset_split + ".pickle")
        if not os.path.exists(save_path):
            with open(save_path, "wb") as save_file:
                pickle.dump(dataset_examples[dataset_split], save_file)
        else:
            print(f"{save_path} exist")
    if not os.path.exists(os.path.join(dataset_path, "to_skip.pickle")):
        with open(os.path.join(dataset_path, "to_skip.pickle"), "wb") as save_file:
            pickle.dump(dataset_filters, save_file)
    else:
        print(f"{os.path.join(dataset_path, 'to_skip.pickle')} exist")


    train_examples = dataset_examples["train"]
    entity_dist_path = os.path.join(dataset_path, "entity_dist.pkl")
    relation_dist_path = os.path.join(dataset_path, "relation_dist.pkl")

    if not os.path.exists(entity_dist_path):
        entity_set = set(np.concatenate((train_examples[:,0], train_examples[:,2])))
        entity_count = []
        for entity in tqdm(entity_set):
            entity_count.append(((train_examples[:,0] == entity) | (train_examples[:,2] == entity)).sum().item())
        entity_dist = pd.DataFrame({"entities":list(entity_set), "count":entity_count})
        entity_dist = entity_dist.sort_values(by="count", ascending=False)
        entity_dist.to_pickle(entity_dist_path)
    else:
        print(f"{entity_dist_path} exist")

    if not os.path.exists(relation_dist_path):
        relation_set = set(train_examples[:,1])
        relation_count = []
        for relation in tqdm(relation_set):
            relation_count.append((train_examples[:,1] == relation).sum().item())
        relation_dist = pd.DataFrame({"relation":list(relation_set), "count":relation_count})
        relation_dist = relation_dist.sort_values(by="count", ascending=False)
        relation_dist.to_pickle(relation_dist_path)
    else:
        print(f"{relation_dist_path} exist")

if __name__ == "__main__":
    dataset_path = "./data/Nations_copy"
    dataset_preprocessing(dataset_path)



# %%
