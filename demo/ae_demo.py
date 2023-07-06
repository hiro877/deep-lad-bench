#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

import numpy as np

sys.path.append("../")
import argparse
import logging
from torch.utils.data import DataLoader
from pytorch_memlab import MemReporter
from collections import defaultdict

from deeploglizer.models import AutoEncoder
from deeploglizer.common.preprocess import FeatureExtractor
from deeploglizer.common.dataloader import load_sessions, log_dataset, log_dataset_gen
from deeploglizer.common.utils import seed_everything, dump_params, dump_final_results


parser = argparse.ArgumentParser()

##### Model params
parser.add_argument("--model_name", default="Autoencoder", type=str)
parser.add_argument("--hidden_size", default=128, type=int)
parser.add_argument("--num_directions", default=2, type=int)
parser.add_argument("--num_layers", default=2, type=int)
parser.add_argument("--embedding_dim", default=32, type=int)

##### Dataset params
parser.add_argument("--dataset", default="HDFS", type=str)
parser.add_argument(
    "--data_dir", default="../data/processed/HDFS_100k/hdfs_0.0_tar", type=str
)
parser.add_argument("--window_size", default=10, type=int)
parser.add_argument("--stride", default=1, type=int)
parser.add_argument("--data_pct", default=1.0, type=float)


##### Input params
parser.add_argument("--feature_type", default="sequentials", type=str, choices=["sequentials", "semantics"])
parser.add_argument("--use_tfidf", action="store_true")
parser.add_argument("--max_token_len", default=50, type=int)
parser.add_argument("--min_token_count", default=1, type=int)
# Uncomment the following to use pretrained word embeddings. The "embedding_dim" should be set as 300
# parser.add_argument(
#     "--pretrain_path", default="../data/pretrain/wiki-news-300d-1M.vec", type=str
# )

##### Training params
parser.add_argument("--epoches", default=100, type=int)
parser.add_argument("--batch_size", default=1024, type=int)
parser.add_argument("--test_batch_size", default=4096, type=int)
parser.add_argument("--learning_rate", default=0.01, type=float)
parser.add_argument("--anomaly_ratio", default=0.1, type=float)
parser.add_argument("--patience", default=3, type=int)

##### Others
parser.add_argument("--random_seed", default=42, type=int)
parser.add_argument("--gpu", default=0, type=int)
parser.add_argument("--cache", default=False, type=bool)
parser.add_argument("--n_experiment", default=5, type=int)


params = vars(parser.parse_args())



if __name__ == "__main__":
    seed_everything(params["random_seed"])

    session_train, session_test = load_sessions(data_dir=params["data_dir"])
    ext = FeatureExtractor(**params)

    session_train = ext.fit_transform(session_train)
    session_test = ext.transform(session_test, datatype="test")

    dataset_test = log_dataset(
        session_test, feature_type=params["feature_type"])
    dataloader_test = DataLoader(
        dataset_test, batch_size=params["test_batch_size"], shuffle=True, pin_memory=True)

    # for pct in [0.1 * (i + 1) for i in range(10)]:
    for pct in [1.0]:
        params["data_pct"] = pct
        model_save_path = dump_params(params)

        logging.info("{}Experiment data_pct:{}%{}".format("=" * 15, pct, "=" * 15))
        dataset_train = log_dataset(
            session_train, feature_type=params["feature_type"], data_pct=pct, shuffle=True)
        dataloader_train = DataLoader(
            dataset_train, batch_size=params["batch_size"], shuffle=True, pin_memory=True)

        store_dict = defaultdict(list)
        best_f1 = -float("inf")
        for i in range(params["n_experiment"]):
            logging.info("{}Experiment #{}{}".format("-" * 15, i, "-" * 15))

            model = AutoEncoder(
                meta_data=ext.meta_data, model_save_path=model_save_path, **params
            )
            reporter = MemReporter(model)
            reporter.report()

            eval_results = model.fit(
                dataloader_train,
                test_loader=dataloader_test,
                epoches=params["epoches"],
                learning_rate=params["learning_rate"],
            )
            reporter.report()

            logging.info("-" * 40)

            for k, v in eval_results.items():
                store_dict[k].append(v)

            if eval_results["f1"] > best_f1:
                best_f1 = eval_results["f1"]
                model.save_model()

        whole_results = defaultdict(float)
        for k, v in store_dict.items():
            mean_score = np.mean(store_dict[k])
            whole_results[k] = mean_score

        result_str = "Final Result: " + \
                     "\t".join(["{}-{:.4f}".format(k, v) for k, v in whole_results.items()])
        logging.info(result_str)
        logging.info("=" * 40)

        dump_final_results(params, whole_results, model)