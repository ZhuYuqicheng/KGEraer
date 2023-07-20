#%%
import argparse
import os

from KGUnlearn import model_test

#%%
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

    # deletion_mode, n_del, degree_set, repeat
    # run_influence(args, "random", 1, repeat=2, save_model=True)
    # run_influence(args, "degree", 1, degree_set={"entity": ["low", 1], "relation": ["low", 1]}, repeat=2, save_model=True)
    model = model_test(args)

# %%
