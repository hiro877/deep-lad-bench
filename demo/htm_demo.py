#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

sys.path.append("../")
import argparse
import logging
import numpy as np
import pandas as pd

from collections import defaultdict

from deeploglizer.models import HTM
from deeploglizer.common.preprocess import FeatureExtractor
from deeploglizer.common.dataloader import load_sessions, log_dataset
from deeploglizer.common.utils import seed_everything, dump_final_results, dump_params


parser = argparse.ArgumentParser()

##### Model params
parser.add_argument("--model_name", default="HTM", type=str)
# enc
parser.add_argument("--enc_sparsity", default=0.15, type=float)
# sp
parser.add_argument("--use_spatial", action="store_true")
parser.add_argument("--column_size", default=800, type=int)
parser.add_argument("--potential_radius", default=500, type=int)
parser.add_argument("--potential_pct", default=0.9, type=float)
parser.add_argument("--local_density", default=0.05, type=float)
# tm
parser.add_argument("--use_temporal", action="store_true")
parser.add_argument("--num_cells", default=8, type=int)
parser.add_argument("--act_threshold", default=11, type=int)
parser.add_argument("--max_segments", default=161, type=int)
parser.add_argument("--max_synapses", default=48, type=int)
parser.add_argument("--min_threshold", default=8, type=int)

##### Dataset params
parser.add_argument("--dataset", default="HDFS", type=str)
parser.add_argument(
    "--data_dir", default="../data/processed/HDFS_100k/hdfs_0.0_tar", type=str
)
parser.add_argument("--truncate_size", default=20, type=int)
parser.add_argument("--pad_size", default=20, type=int)

##### Input params
parser.add_argument("--feature_type", default="sequentials", type=str, choices=["sequentials", "semantics"])
parser.add_argument("--window_type", default="session", type=str, choices=["sliding", "session"])
parser.add_argument("--label_type", default="anomaly", type=str)
parser.add_argument("--use_tfidf", action="store_true")
parser.add_argument("--max_token_len", default=50, type=int)
parser.add_argument("--min_token_count", default=1, type=int)

##### Training params
# sp
parser.add_argument("--boost_strength", default=3.5, type=float)
parser.add_argument("--syn_inc", default=0.03, type=float)
parser.add_argument("--syn_connect", default=0.14, type=float)
parser.add_argument("--syn_dec", default=0.001, type=float)
# tm
parser.add_argument("--init_perm", default=0.22, type=float)
parser.add_argument("--perm_dec", default=0.08, type=float)
parser.add_argument("--perm_inc", default=0.08, type=float)
parser.add_argument("--new_synapse_count", default=18, type=int)

##### Others
parser.add_argument("--random_seed", default=42, type=int)
parser.add_argument("--n_experiment", default=5, type=int)

params = vars(parser.parse_args())



def stat_dataset(dataset):
    # stat_dataset(dataset_test)
    l = np.array([len(x["features"]) for x in dataset])
    print(pd.DataFrame(pd.Series(l.ravel()).describe()).transpose())

def viz_data(dataset, column="features"):
    # viz_data(dataset_test, column="features")
    for d in dataset:
        plt.scatter(np.arange(len(d[column])), d[column])
        plt.show()
        plt.close()



if __name__ == "__main__":
    seed_everything(params["random_seed"])

    session_train, session_test = load_sessions(data_dir=params["data_dir"])

    ext = FeatureExtractor(**params)

    session_train = ext.fit_transform(session_train)
    session_test = ext.transform(session_test, datatype="test")
    dataset_test = log_dataset(
        session_test, feature_type=params["feature_type"])

    # for pct in [0.1 * (i+1) for i in range(10)]:
    for pct in [1.0]:
        params["data_pct"] = pct
        model_save_path = dump_params(params)

        logging.info("{}Experiment data_pct:{}%{}".format("=" * 15, pct, "=" * 15))
        dataset_train = log_dataset(
            session_train, feature_type=params["feature_type"], data_pct=pct, shuffle=True)

        store_dict = defaultdict(list)
        best_f1 = -float("inf")
        for i in range(params["n_experiment"]):
            logging.info("{}Experiment #{}{}".format("-" * 15, i, "-" * 15))

            model = HTM(meta_data=ext.meta_data, model_save_path=model_save_path, **params)

            eval_results = model.fit(
                dataset_train=dataset_train,
                dataset_test=dataset_test
            )

            logging.info("-" * 40)

            for k, v in eval_results.items():
                store_dict[k].append(v)

            if eval_results["f1"] > best_f1:
                best_f1 = eval_results["f1"]
                model.save_model()

        model.load_model(model_save_path=model_save_path)
        whole_results = defaultdict(float)
        for k, v in store_dict.items():
            mean_score = np.mean(store_dict[k])
            whole_results[k] = mean_score

        result_str = "Final Result: " + \
                     "\t".join(["{}-{:.4f}".format(k, v) for k, v in whole_results.items()])
        logging.info(result_str)
        logging.info("=" * 40)

        dump_final_results(params, whole_results, model)