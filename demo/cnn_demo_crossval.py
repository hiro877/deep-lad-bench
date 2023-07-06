#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import os
import sys

import numpy as np

sys.path.append("../")
import argparse
from torch.utils.data import DataLoader
from pytorch_memlab import MemReporter
from collections import defaultdict

from deeploglizer.models import CNN
from deeploglizer.common.dataloader import load_sessions, log_dataset, log_dataset_gen
from deeploglizer.common.preprocess import FeatureExtractor
from deeploglizer.common.utils import seed_everything, dump_final_results, dump_params, dump_store_results
import csv


parser = argparse.ArgumentParser()

##### Model params
parser.add_argument("--model_name", default="CNN", type=str)
parser.add_argument("--hidden_size", default=128, type=int)
parser.add_argument("--kernel_sizes", default="2 3 4", nargs="+")
parser.add_argument("--embedding_dim", default=32, type=int)

##### Dataset params
parser.add_argument("--dataset", default="HDFS", type=str)
parser.add_argument(
    "--data_dir", default="../data/processed/HDFS_100k/hdfs_1.0_tar", type=str
)
parser.add_argument("--window_size", default=10, type=int)
parser.add_argument("--stride", default=1, type=int)
parser.add_argument("--data_pct", default=1.0, type=float)


##### Input params
parser.add_argument("--feature_type", default="sequentials", type=str, choices=["sequentials", "semantics"])
parser.add_argument("--label_type", default="anomaly", type=str)
parser.add_argument("--use_tfidf", action="store_true")
parser.add_argument("--pretrain_path", default=None, type=str)
parser.add_argument("--max_token_len", default=50, type=int)
parser.add_argument("--min_token_count", default=1, type=int)
parser.add_argument("--eval_type", default="window", type=str)
# Uncomment the following to use pretrained word embeddings. The "embedding_dim" should be set as 300
# parser.add_argument(
#     "--pretrain_path", default="../data/pretrain/wiki-news-300d-1M.vec", type=str
# )

##### Training params
parser.add_argument("--epoches", default=100, type=int)
parser.add_argument("--batch_size", default=1024, type=int)
parser.add_argument("--test_batch_size", default=4096, type=int)
parser.add_argument("--learning_rate", default=0.01, type=float)
parser.add_argument("--patience", default=3, type=int)

##### Others
parser.add_argument("--random_seed", default=42, type=int)
parser.add_argument("--gpu", default=0, type=int)
parser.add_argument("--n_experiment", default=5, type=int)
parser.add_argument("--is_validation", action='store_true')
parser.add_argument("--is_comparing_val", action='store_true')
parser.add_argument("--use_val_template_in_learning", action='store_true')
parser.add_argument("--wtp_records_path", default="wtp_records", type=str)


params = vars(parser.parse_args())



if __name__ == "__main__":
    is_shuffle = True
    use_all_template_in_learning = False
    print("is_shuffle: ", is_shuffle)
    print("use_all_template_in_learning: ", use_all_template_in_learning)
    seed_everything(params["random_seed"])
    # session_train, session_test, _ = load_sessions(data_dir=params["data_dir"], is_validation=params["is_validation"])
    session_train, session_test, session_val = load_sessions(data_dir=params["data_dir"], is_validation=params["is_validation"])
    ext = FeatureExtractor(**params)

    ext.is_print_dataset=True
    session_train = ext.fit_transform(session_train)

    # wtp_window_labels={}
    # wtp_window_anomalies={}
    # wtp_info_all={"anomalies":[], "labels":[], "wtp":[]}
    dataset_train = log_dataset(
        session_train, feature_type=params["feature_type"], data_pct=1, shuffle=is_shuffle)
    dataloader_train = DataLoader(
        dataset_train, batch_size=params["batch_size"], shuffle=is_shuffle, pin_memory=True)

    os.makedirs(params["wtp_records_path"], exist_ok=True)
    # with open('wtp_records/window_template_patterns_train.csv', 'w') as f:
    #     writer = csv.writer(f)
    #     for k, v in ext.wtp_tr.items():
    #         writer.writerow([k, v, wtp_window_labels[k], wtp_window_anomalies[k]])

    if params["is_validation"]:
        if params["use_val_template_in_learning"]:
            ext.update_id2log_train(session_val)
        session_val = ext.transform(session_val, datatype="test")
        dataset_val = log_dataset(
            session_val, feature_type=params["feature_type"])
        dataloader_val = DataLoader(
            dataset_val, batch_size=params["test_batch_size"], shuffle=is_shuffle, pin_memory=True)

    if use_all_template_in_learning:
        ext.update_id2log_train(session_test)
    session_test = ext.transform(session_test, datatype="test")
    # sys.exit()
    dataset_test = log_dataset(
        session_test, feature_type=params["feature_type"])

    """ Check Validation Size """
    # print(len(dataset_train))
    # validation_ratio = 0.11
    # dataset_val = dataset_train[-int(len(dataset_train)*validation_ratio):]
    # dataset_train = dataset_train[:-int(len(dataset_train)*validation_ratio)]
    # print("len(dataset_val): ", len(dataset_val))
    # print("len(dataset_train): ", len(dataset_train))
    # print("len(dataset_test): ", len(dataset_test))
    # sys.exit()

    dataloader_test = DataLoader(
        dataset_test, batch_size=params["test_batch_size"], shuffle=is_shuffle, pin_memory=True)

    # with open('wtp_records/window_template_patterns_test.csv', 'w') as f:
    #     writer = csv.writer(f)
    #     for k, v in ext.wtp_te.items():
    #         writer.writerow([k, v, wtp_window_labels[k], wtp_window_anomalies[k]])
    #         if ext.wtp_tr.get(k):
    #             ext.wtp_tr[k]+=v
    #         else:
    #             ext.wtp_tr[k]=1


    # with open('window_template_patterns_all.csv', 'w') as f:
    #     writer = csv.writer(f)
    #     for k, v in ext.wtp_tr.items():
    #         writer.writerow([k, v, wtp_window_labels[k], wtp_window_anomalies[k]])

    # if params["is_validation"]:
    #     session_val = ext.transform(session_val, datatype="test")
    #     dataset_val = log_dataset(
    #         session_val, feature_type=params["feature_type"])
    #     dataloader_val = DataLoader(
    #         dataset_val, batch_size=params["test_batch_size"], shuffle=is_shuffle, pin_memory=True)


    # for pct in [0.1 * (i + 1) for i in range(10)]:
    for pct in [1]:
        params["data_pct"] = pct
        model_save_path = dump_params(params)

        logging.info("{}Experiment data_pct:{}%{}".format("=" * 15, pct, "=" * 15))
        # dataset_train = log_dataset(
        #     session_train, feature_type=params["feature_type"], data_pct=pct, shuffle=is_shuffle)

        if params["is_validation"]:
            print("len(dataset_val): ", len(dataset_val))
            print("len(dataset_train): ", len(dataset_train))
            print("len(dataset_test): ", len(dataset_test))


        elif params["is_comparing_val"]:
            print("len(dataset_val): ", len(dataset_val))
            print("len(dataset_train): ", len(dataset_train))
            print("len(dataset_test): ", len(dataset_test))

            # dataloader_train = DataLoader(
            #     dataset_train, batch_size=params["batch_size"], shuffle=is_shuffle, pin_memory=True)

        else:
            print("len(dataset_train): ", len(dataset_train))
            print("len(dataset_test): ", len(dataset_test))
            # dataloader_train = DataLoader(
            #     dataset_train, batch_size=params["batch_size"], shuffle=is_shuffle, pin_memory=True)

        store_dict = defaultdict(list)
        best_f1 = -float("inf")
        for i in range(params["n_experiment"]):
            # model.exp_analisys_dict = {"result": [], "miss_pred": {}, "miss_pred_wtp": {}}
            logging.info("{}Experiment #{}{}".format("-" * 15, i, "-" * 15))

            model = CNN(meta_data=ext.meta_data, model_save_path=model_save_path, **params)
            model.is_learning_curve=True
            model.wtp_records_path = params["wtp_records_path"]
            # reporter = MemReporter(model)
            # reporter.report()

            if params["is_validation"]:
                eval_results = model.fit(
                    dataloader_train,
                    test_loader=dataloader_val,
                    epoches=params["epoches"],
                    learning_rate=params["learning_rate"],
                )
            else:
                eval_results = model.fit(
                    dataloader_train,
                    test_loader=dataloader_test,
                    epoches=params["epoches"],
                    learning_rate=params["learning_rate"],
                )
            # reporter.report()

            logging.info("-" * 40)

            if params["is_validation"]:
                # logging.info("##############################")
                # logging.info("TestLoader inference. ")
                # model.is_exp_analisys=True
                eval_results = model.evaluate(dataloader_test)
                # model.is_exp_analisys = False
                # logging.info("##############################")

                # with open('wtp_records/miss_pred_exp{}.csv'.format(i), 'w') as f:
                #     writer = csv.writer(f)
                #     for k, v in model.exp_analisys_dict["miss_pred"].items():
                #         writer.writerow([k, v, wtp_window_labels[v], wtp_window_labels[v]])
                # with open('wtp_records/miss_pred_wtp_exp{}.csv'.format(i), 'w') as f:
                #     writer = csv.writer(f)
                #     for k, v in model.exp_analisys_dict["miss_pred_wtp"].items():
                #         writer.writerow([k, v, wtp_window_labels[k], wtp_window_labels[k]])


            if eval_results["f1"] > best_f1:
                best_f1 = eval_results["f1"]
                model.save_model()
                print(best_f1)
                print(eval_results)
                print("===== End Calculate Best_Fa =====")

            for k, v in eval_results.items():
                store_dict[k].append(v)

        # with open('wtp_records/window_template_patterns_all.csv', 'w') as f:
        #     writer = csv.writer(f)
        #     i=0
        #     for k, v in ext.wtp_tr.items():
        #         # writer.writerow([k, v, wtp_window_labels[k], wtp_window_anomalies[k], model.exp_analisys_dict["result"][i]])
        #         writer.writerow([k, v, wtp_window_labels[k], wtp_window_anomalies[k]])
        #     i+=1
        # with open('wtp_records/log2id_train.csv'.format(i), 'w') as f:
        #     writer = csv.writer(f)
        #     for k, v in ext.log2id_train.items():
        #         writer.writerow([k, v])

        # if wtp_info_all["wtp"]:
        #     print("save wtp_info_all.csv")
        #     print(len(wtp_info_all["anomalies"]), len(wtp_info_all["labels"]), len(wtp_info_all["wtp"]))
        #     print(len(ext.wtp_info_all["templates"]))
        #     with open('wtp_records/wtp_info_all.csv'.format(i), 'w') as f:
        #         writer = csv.writer(f)
        #         for index in range(len(wtp_info_all["anomalies"]),):
        #             a=[wtp_info_all["wtp"][index], wtp_info_all["anomalies"][index]] + ext.wtp_info_all["templates"][index]
        #             writer.writerow(a)


        whole_results = defaultdict(float)
        for k, v in store_dict.items():
            mean_score = np.mean(store_dict[k])
            whole_results[k] = mean_score

        result_str = "Final Result: " + \
                     "\t".join(["{}-{:.4f}".format(k, v) for k, v in whole_results.items()])
        logging.info(result_str)
        logging.info("=" * 40)

        dump_final_results(params, whole_results, model)
        if params["is_validation"]:
            dump_store_results("BGL_VAL_Store", store_dict)
        elif params["is_comparing_val"]:
            dump_store_results("BGL_CMPVAL_Store", store_dict)
        else:
            dump_store_results("BGL_Store", store_dict)