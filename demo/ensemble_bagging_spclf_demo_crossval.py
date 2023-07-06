#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append("../")
import argparse
import logging
import numpy as np

from collections import defaultdict

from deeploglizer.models import SPClassifier, SPClassifierProcess
from deeploglizer.common.preprocess import FeatureExtractor
from deeploglizer.common.dataloader import load_sessions, log_dataset, log_dataset_no_duplicates
from deeploglizer.common.utils import seed_everything, dump_final_results, dump_params, dump_store_results, dump_store_results_cm

import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
import threading # デバッグ用
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, roc_curve, roc_auc_score
import copy
import random
import matplotlib.pyplot as plt

def saveDataset(name, ext, dataset_train, dataset_test, dataset_val=None):
    save_path = "/work2/huchida/Datasets/BGL"
    with open(save_path+"/"+name+"/dataset_train.pkl", "wb") as file:
        pickle.dump(dataset_train, file)

    with open(save_path+"/"+name+"/dataset_test.pkl", "wb") as file:
        pickle.dump(dataset_test, file)

    if dataset_val:
        with open(save_path+"/"+name+"/dataset_val.pkl", "wb") as file:
            pickle.dump(dataset_val, file)

    with open(save_path+"/"+name+"/ext.pkl", "wb") as file:
        pickle.dump(ext, file)


def loadDataset(name):
    save_path = "/work2/huchida/Datasets/BGL"
    dataset_val = None
    with open(save_path + "/" + name + "/dataset_train.pkl", "rb") as file:
        dataset_train = pickle.load(file)

    with open(save_path + "/" + name + "/dataset_test.pkl", "rb") as file:
        dataset_test = pickle.load(file)

    if os.path.exists(save_path + "/" + name + "/dataset_val.pkl"):
        with open(save_path + "/" + name + "/dataset_val.pkl", "rb") as file:
            dataset_val = pickle.load(file)

    with open(save_path + "/" + name + "/ext.pkl", "rb") as file:
        ext = pickle.load(file)

    return ext, dataset_train, dataset_test, dataset_val

def do_multiprocess(process_num, src_datas):
    datas = []
    with ProcessPoolExecutor(max_workers=process_num) as executor:
        results = executor.map(
            fit, src_datas, timeout=None)
        for data in results:
            datas.append(data)
    return datas

# def fit(model, model_num, dataset_train, dataset_test, dataset_val):
def fit(args):
    """自作関数です。並列で実行します。"""
    # print(args)
    model, model_num, dataset_train, dataset_test, dataset_val = args
    print(f'実行中 jisaku_func({model_num})')

    # (デバッグ) 親プロセス ID (PPID) を取得してみます。
    ppid = os.getppid()

    # (デバッグ) プロセス ID (PID) を取得してみます。
    pid = os.getpid()

    # (デバッグ) スレッド ID (TID) を取得してみます。
    tid = threading.get_native_id() # Python 3.8 から使用可能
    print(model_num, ppid, pid, tid)

    if params["is_validation"]:
        # if params["is_epoches"]:
        #     eval_results = model.fit_epoches(
        #         dataset_train=dataset_train,
        #         dataset_test=dataset_val
        #     )
        # else:
        #     eval_results = model.fit(
        #         dataset_train=dataset_train,
        #         dataset_test=dataset_val
        #     )
        eval_results = model.fit(
            dataset_train=dataset_train,
            dataset_test=dataset_val
        )
    else:
        # if params["is_epoches"]:
        #     eval_results = model.fit_epoches(
        #         dataset_train=dataset_train,
        #         dataset_test=dataset_test
        #     )
        # else:
        #     eval_results = model.fit(
        #         dataset_train=dataset_train,
        #         dataset_test=dataset_test
        #     )
        eval_results = model.fit(
            dataset_train=dataset_train,
            dataset_test=dataset_test
        )
    eval_results_for_val = None
    if params["is_validation"]:
        logging.info('\033[91m' + "##############################" + '\033[0m')
        logging.info("TestDataset inference. ")
        eval_results_for_val = model.evaluate_process(dataset_test)
        logging.info('\033[91m' + "##############################" + '\033[0m')
    return (eval_results, eval_results_for_val)


def evaluate_multiprocess(results, dataset_test, is_validation=False):
    print("="*20)
    print("evaluate_multiprocess()")
    print("=" * 20)
    results_len = len(results)
    result_num = 0
    if is_validation:
        result_num = 1


    y_true = []
    y_pred = []
    y_prob = []
    print("X"*10)
    for i, d in enumerate(dataset_test):
        feat, lbl = d["features"], d["window_anomalies"]
        y_true.append(lbl)

        mean = 0
        for result in results:
            mean = mean + result[result_num]["y_pred"][i]
        mean = mean / results_len
        # print(mean)
        y_prob.append(mean)
        if(mean > 0.5):
            y_pred.append(1)
        else:
            y_pred.append(0)
    print("X" * 10)
    best_f1 = -1
    best_model_num = 0
    """ For Selecting Best Process Model """
    for i, result in enumerate(results):
        eval_results = {
            "f1": f1_score(y_true, result[result_num]["y_pred"]),
            "rc": recall_score(y_true, result[result_num]["y_pred"]),
            "pc": precision_score(y_true, result[result_num]["y_pred"]),
            "acc": accuracy_score(y_true, result[result_num]["y_pred"]),
        }
        print("process{}: ".format(i), eval_results)
        if eval_results["f1"] > best_f1:
            best_f1 = eval_results["f1"]
            best_model_num = i
            # print("best_f1={}, best_model_num={}".format(best_f1, best_model_num))
    print("="*20)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.flatten()
    eval_results = {
        "f1": f1_score(y_true, y_pred),
        "rc": recall_score(y_true, y_pred),
        "pc": precision_score(y_true, y_pred),
        "acc": accuracy_score(y_true, y_pred),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "roc_auc": roc_auc_score(y_true, y_prob)
    }
    print("evaluate_multiprocess(): eval_results=", eval_results)

    # ROC曲線を描写
    # fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    # plt.plot(fpr, tpr, label='spclf')
    # plt.fill_between(fpr, tpr, 0, alpha=0.1)
    #
    #
    # plt.xlabel('FPR')
    # plt.ylabel('TPR')
    # plt.legend()
    # plt.savefig("ROC.png")  # プロットしたグラフをファイルsin.pngに保存する
    # plt.show()

    # AUCの計算
    # print(f'SPCLF AUR: {roc_auc_score(y_true, y_prob):.4f}')
    # print("fpr={}, tpr={}, thresholds={}".format(fpr, tpr, thresholds))
    return eval_results, best_model_num

def evaluate_multiprocess_save_roc(results, dataset_test, exp_num, exp_str, is_validation=False):
    print("="*20)
    print("evaluate_multiprocess()")
    print("=" * 20)
    results_len = len(results)
    result_num = 0
    if is_validation:
        result_num = 1


    y_true = []
    y_pred = []
    y_prob = []
    for i, d in enumerate(dataset_test):
        feat, lbl = d["features"], d["window_anomalies"]
        y_true.append(lbl)

        mean = 0
        for result in results:
            mean = mean + result[result_num]["y_pred"][i]
        mean = mean / results_len
        y_prob.append(mean)
        if(mean > 0.5):
            y_pred.append(1)
        else:
            y_pred.append(0)
    best_f1 = -1
    best_model_num = 0
    """ For Selecting Best Process Model """
    for i, result in enumerate(results):
        eval_results = {
            "f1": f1_score(y_true, result[result_num]["y_pred"]),
            "rc": recall_score(y_true, result[result_num]["y_pred"]),
            "pc": precision_score(y_true, result[result_num]["y_pred"]),
            "acc": accuracy_score(y_true, result[result_num]["y_pred"]),
        }
        print("process{}: ".format(i), eval_results)
        if eval_results["f1"] > best_f1:
            best_f1 = eval_results["f1"]
            best_model_num = i
            # print("best_f1={}, best_model_num={}".format(best_f1, best_model_num))
    print("="*20)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.flatten()
    eval_results = {
        "f1": f1_score(y_true, y_pred),
        "rc": recall_score(y_true, y_pred),
        "pc": precision_score(y_true, y_pred),
        "acc": accuracy_score(y_true, y_pred),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "roc_auc": roc_auc_score(y_true, y_prob)
    }
    print("evaluate_multiprocess(): eval_results=", eval_results)

    # ROC曲線を描写
    os.makedirs("Results_ROC/"+exp_str+"/", exist_ok=True)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    plt.plot(fpr, tpr, label='spclf')
    plt.fill_between(fpr, tpr, 0, alpha=0.1)

    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.savefig("Results_ROC/"+exp_str+"/"+"/ROC{}.png".format(exp_num))  # プロットしたグラフをファイルsin.pngに保存する

    return eval_results, best_model_num

def evaluate_multiprocess_with_vote(results, dataset_test):
    print("="*20)
    print("evaluate_multiprocess()")
    print("=" * 20)
    results_num = len(results)

    y_true = []
    y_pred = []
    total_vote = [0 for i in range(process_num)]
    for i, d in enumerate(dataset_test):
        feat, lbl = d["features"], d["window_anomalies"]
        y_true.append(lbl)

        voted_true_indexes = []
        voted_false_indexes = []
        for j, result in enumerate(results):
            # mean = mean + result["y_pred"][i]
            if result["y_pred"][i] == 1:
                voted_true_indexes.append(j)
            else:
                voted_false_indexes.append(j)
        if len(voted_true_indexes) > len(voted_false_indexes):
            y_pred.append(1)
            # 投票者の中で一番投票回数が多いものをBEST MODELとする
            for index in voted_true_indexes:
                total_vote[index] += 1
        else:
            y_pred.append(0)
            for index in voted_false_indexes:
                total_vote[index] += 1
        # mean = mean / results_num
        # if(mean > 0.5):
        #     y_pred.append(1)
        # else:
        #     y_pred.append(0)
    best_f1 = -1;
    for i, result in enumerate(results):
        eval_results = {
            "f1": f1_score(y_true, result["y_pred"]),
            "rc": recall_score(y_true, result["y_pred"]),
            "pc": precision_score(y_true, result["y_pred"]),
            "acc": accuracy_score(y_true, result["y_pred"]),
        }
        print("process{}: ".format(i), eval_results)
        if eval_results["f1"] > best_f1:
            best_f1 = eval_results["f1"]
            print("best_f1={}".format(best_f1))
    print("="*20)

    best_voted = -1
    best_model_num = 0
    for i, vote in enumerate(total_vote):
        if vote > best_voted:
            best_voted = vote
            best_model_num = i
    # print("best_model_num={}".format(best_model_num))

    # eval_results = {
    #     "f1": f1_score(y_true, y_pred),
    #     "rc": recall_score(y_true, y_pred),
    #     "pc": precision_score(y_true, y_pred),
    #     "acc": accuracy_score(y_true, y_pred),
    # }
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.flatten()
    eval_results = {
        "f1": f1_score(y_true, y_pred),
        "rc": recall_score(y_true, y_pred),
        "pc": precision_score(y_true, y_pred),
        "acc": accuracy_score(y_true, y_pred),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }
    print(eval_results)
    print("=" * 20)
    print("evaluate_multiprocess END")
    print("="*20)
    return eval_results, best_model_num

# def make_learning_curve(eval_results, wtp_records_path):
#     with open(wtp_records_path + '/Learning_Curve.csv', 'a') as f:
#         writer = csv.writer(f)
#         writer.writerow(["best_rersult", str(eval_results)])
#         writer.writerow(["train"])
#         writer.writerow(["epoch", "f1", "rc", "pc", "acc"])
#         for index in range(len(lc_datas_train)):
#             writer.writerow(lc_datas_train[index])
#         writer.writerow(["test"])
#         writer.writerow(["epoch", "f1", "rc", "pc", "acc"])
#         for index in range(len(lc_datas_test)):
#             writer.writerow(lc_datas_test[index])

def a():
    return ({"a":1}, {"a":2})


parser = argparse.ArgumentParser()

##### Model params
parser.add_argument("--model_name", default="SPClassifier", type=str)

parser.add_argument("--emb_dim", default=500, type=int)
parser.add_argument("--emb_sparsity", default=0.15, type=float)

parser.add_argument("--potential_radius", default=7, type=int)
parser.add_argument("--duty_cycle", default=1402, type=int)
parser.add_argument("--stimulus_thresh", default=6, type=int)
parser.add_argument("--potential_pct", default=0.1, type=float)
parser.add_argument("--local_density", default=0.2, type=float)
parser.add_argument("--boost_strength", default=7.0, type=float)
parser.add_argument("--syn_inc", default=0.14, type=float)
parser.add_argument("--syn_connect", default=0.5, type=float)
parser.add_argument("--syn_dec", default=0.02, type=float)
parser.add_argument("--min_overlap_duty", default=0.2, type=float)

##### Dataset params
parser.add_argument("--dataset", default="HDFS", type=str)
parser.add_argument(
    "--data_dir", default="../data/processed/HDFS_100k/hdfs_0.0_tar", type=str
)
parser.add_argument("--window_size", default=10, type=int)
parser.add_argument("--stride", default=1, type=int)

##### Input params
parser.add_argument("--feature_type", default="sequentials", type=str, choices=["sequentials", "semantics"])
parser.add_argument("--label_type", default="anomaly", type=str)
parser.add_argument("--use_tfidf", action="store_true")
parser.add_argument("--max_token_len", default=50, type=int)
parser.add_argument("--min_token_count", default=1, type=int)
parser.add_argument("--eval_type", default="window", type=str)

##### Others
parser.add_argument("--random_seed", default=42, type=int)
parser.add_argument("--n_experiment", default=5, type=int)
parser.add_argument("--epoches", default=100, type=int)
parser.add_argument("--patience", default=3, type=int)
parser.add_argument("--is_validation", action='store_true')
parser.add_argument("--use_val_template_in_learning", action='store_true')
parser.add_argument("--wtp_records_path", default="wtp_records", type=str)
parser.add_argument("--is_epoches", action='store_true')
parser.add_argument("--is_log_dataset_no_duplicates", action='store_true')
parser.add_argument("--use_best_model", action='store_true')
parser.add_argument("--use_best_model_with_vote", action='store_true')
parser.add_argument("--dataset_name", default="tr90a", type=str)
parser.add_argument("--is_split_dataset", action='store_true')
parser.add_argument("--process_num", default=1, type=int)
parser.add_argument("--improved_bagging", action='store_true')

params = vars(parser.parse_args())

def main(params, process_num, src_datas, dataset_test, dataset_val, exp_num):
    if params["is_epoches"]:
        eval_results_epochs = []
        eval_results_epochs_val = []
        for i in range(params["epoches"]):
            datas = do_multiprocess(process_num, src_datas)
            # eval_results , _ = evaluate_multiprocess(datas, dataset_test)
            # eval_results_epochs.append(eval_results)
            if params["is_validation"]:
                eval_results, best_model_num = evaluate_multiprocess(datas, dataset_val)
                eval_results_epochs_val.append(copy.deepcopy(eval_results))

                # eval_results, _ = evaluate_multiprocess(datas, dataset_test, True)
                eval_results, _ = evaluate_multiprocess_save_roc(datas, dataset_test, exp_num, params["dataset_name"], True)
                eval_results_epochs.append(eval_results)
            else:
                eval_results, best_model_num = evaluate_multiprocess(datas, dataset_test)
                eval_results_epochs.append(eval_results)
        i = 1
        for eval_result in eval_results_epochs:
            print("Epoch{}".format(i), eval_result)
            i += 1
        if eval_results_epochs_val:
            # i = 1
            # for eval_result in eval_results_epochs_val:
            #     print("Val: Epoch{}".format(i), eval_result)
            #     i += 1
            dump_store_results_cm("BGL_SPCLF_EvalResult_VAL", eval_results_epochs_val)
    else:
        datas = do_multiprocess(process_num, src_datas)
        if params["is_validation"]:
            eval_results_val, best_model_num = evaluate_multiprocess(datas, dataset_val)

            # eval_results, _ = evaluate_multiprocess(datas, dataset_test, True)
            eval_results, _ = evaluate_multiprocess_save_roc(datas, dataset_test, exp_num, params["dataset_name"], True)
            dump_store_results_cm("BGL_SPCLF_EvalResult_VAL", eval_results_val)
        else:
            eval_results, best_model_num = evaluate_multiprocess(datas, dataset_test)
        # eval_results, _ = evaluate_multiprocess(datas, dataset_test)
    return eval_results
def main_best_model(ext, params, process_num, src_datas, dataset_test, dataset_val, exp_num):
    if params["is_epoches"]:
        eval_results_epochs = []
        eval_results_epochs_val = []
        for i in range(params["epoches"]):
            datas = do_multiprocess(process_num, src_datas)
            # print(datas)
            # print(datas[0])
            # print(len(datas))
            if params["is_validation"]:
                eval_results, best_model_num = evaluate_multiprocess(datas, dataset_val)
                eval_results_epochs_val.append(copy.deepcopy(eval_results))

                # eval_results, _ = evaluate_multiprocess(datas, dataset_test, True)
                eval_results, _ = evaluate_multiprocess_save_roc(datas, dataset_test, exp_num, params["dataset_name"],
                                                                 True)
                eval_results_epochs.append(eval_results)
            else:
                eval_results, best_model_num = evaluate_multiprocess(datas, dataset_test)
                eval_results_epochs.append(eval_results)

            # replace best model
            """ With Load/Save """
            print("best_model_num={}".format(best_model_num))
            src_datas[best_model_num][0].save_model()
            load_model_path = src_datas[best_model_num][0].model_save_dir
            for j in range(process_num):
                model = SPClassifierProcess(meta_data=ext.meta_data, model_save_path=load_model_path, **params)
                model.load_model(load_model_path)
                src_datas[j][0] = copy.deepcopy(model)
            """ With DeepCopy """
            # model = copy.deepcopy(src_datas[best_model_num][0])
            # # load_model_path = src_datas[best_model_num][0].model_save_dir
            # for j in range(process_num):
            #     # model = SPClassifierProcess(meta_data=ext.meta_data, model_save_path=load_model_path, **params)
            #     # model.load_model(load_model_path)
            #     src_datas[j][0] = copy.deepcopy(model)

        i = 1
        for eval_result in eval_results_epochs:
            print("Epoch{}".format(i), eval_result)
            i += 1
        if eval_results_epochs_val:
            # i = 1
            # for eval_result in eval_results_epochs_val:
            #     print("Val: Epoch{}".format(i), eval_result)
            #     i += 1
            dump_store_results_cm("BGL_SPCLF_EvalResult_VAL", eval_results_epochs_val)

    else:
        # datas = do_multiprocess(process_num, src_datas)
        # eval_results, _ = evaluate_multiprocess(datas, dataset_test)
        datas = do_multiprocess(process_num, src_datas)
        if params["is_validation"]:
            eval_results_val, best_model_num = evaluate_multiprocess(datas, dataset_val)

            # eval_results, _ = evaluate_multiprocess(datas, dataset_test, True)
            eval_results, _ = evaluate_multiprocess_save_roc(datas, dataset_test, exp_num, params["dataset_name"], True)
            dump_store_results_cm("BGL_SPCLF_EvalResult_VAL", eval_results_val)
        else:
            eval_results, best_model_num = evaluate_multiprocess(datas, dataset_test)
    return eval_results

def main_best_model_with_vote(ext, params, process_num, src_datas, dataset_test, exp_num):
    if params["is_epoches"]:
        eval_results_epochs = []
        eval_results_epochs_val = []
        for i in range(params["epoches"]):
            datas = do_multiprocess(process_num, src_datas)
            # eval_results, best_model_num = evaluate_multiprocess_with_vote(datas, dataset_test)
            # eval_results_epochs.append(eval_results)
            if params["is_validation"]:
                eval_results, best_model_num = evaluate_multiprocess(datas, dataset_val)
                eval_results_epochs_val.append(copy.deepcopy(eval_results))

                # eval_results, _ = evaluate_multiprocess(datas, dataset_test, True)
                eval_results, _ = evaluate_multiprocess_save_roc(datas, dataset_test, exp_num, params["dataset_name"],
                                                                 True)
                eval_results_epochs.append(eval_results)
            else:
                eval_results, best_model_num = evaluate_multiprocess(datas, dataset_test)
                eval_results_epochs.append(eval_results)

            # replace best model
            """ With Load/Save """
            print("best_model_num={}".format(best_model_num))
            src_datas[best_model_num][0].save_model()
            load_model_path = src_datas[best_model_num][0].model_save_dir
            for j in range(process_num):
                model = SPClassifierProcess(meta_data=ext.meta_data, model_save_path=load_model_path, **params)
                model.load_model(load_model_path)
                src_datas[j][0] = copy.deepcopy(model)
            """ With DeepCopy """
            # model = copy.deepcopy(src_datas[best_model_num][0])
            # # load_model_path = src_datas[best_model_num][0].model_save_dir
            # for j in range(process_num):
            #     # model = SPClassifierProcess(meta_data=ext.meta_data, model_save_path=load_model_path, **params)
            #     # model.load_model(load_model_path)
            #     src_datas[j][0] = copy.deepcopy(model)

        i = 1
        for eval_result in eval_results_epochs:
            print("Epoch{}".format(i), eval_result)
            i += 1
        if eval_results_epochs_val:
            # i = 1
            # for eval_result in eval_results_epochs_val:
            #     print("Val: Epoch{}".format(i), eval_result)
            #     i += 1
            dump_store_results_cm("BGL_SPCLF_EvalResult_VAL", eval_results_epochs_val)
    else:
        # datas = do_multiprocess(process_num, src_datas)
        # eval_results, _ = evaluate_multiprocess(datas, dataset_test)
        datas = do_multiprocess(process_num, src_datas)
        if params["is_validation"]:
            eval_results_val, best_model_num = evaluate_multiprocess(datas, dataset_val)

            # eval_results, _ = evaluate_multiprocess(datas, dataset_test, True)
            eval_results, _ = evaluate_multiprocess_save_roc(datas, dataset_test, exp_num, params["dataset_name"], True)
            dump_store_results_cm("BGL_SPCLF_EvalResult_VAL", eval_results_val)
        else:
            eval_results, best_model_num = evaluate_multiprocess(datas, dataset_test)
    return eval_results

if __name__ == '__main__':
    # print(a()[0])
    # sys.exit()
    is_shuffle = True
    use_all_template_in_learning = False
    print("is_shuffle: ", is_shuffle)
    print("use_all_template_in_learning: ", use_all_template_in_learning)
    seed_everything(params["random_seed"])
    process_num = params["process_num"]

    save_path = "/work2/huchida/Datasets/BGL/"
    dataset_val = None
    # name="tr90d_val"
    name=params["dataset_name"]
    print(save_path+name)
    os.makedirs(save_path+name, exist_ok=True)
    if os.path.exists(save_path + "/" + name + "/dataset_train.pkl"):
        ext, dataset_train, dataset_test, dataset_val = loadDataset(name)
    else:
        session_train, session_test, session_val = load_sessions(data_dir=params["data_dir"], is_validation=params["is_validation"])

        ext = FeatureExtractor(**params)
        ext.is_print_dataset=True
        session_train = ext.fit_transform(session_train)

        """ Default """
        # dataset_train = log_dataset(
        #     session_train, feature_type=params["feature_type"], data_pct=1, shuffle=is_shuffle)
        if params["is_log_dataset_no_duplicates"]:
            print("log_dataset_no_duplicates()")
            dataset_train = log_dataset_no_duplicates(
                session_train, feature_type=params["feature_type"], data_pct=1, shuffle=is_shuffle)
        else:
            print("log_dataset()")
            dataset_train = log_dataset(
                session_train, feature_type=params["feature_type"], data_pct=1, shuffle=is_shuffle)
        # dataset_train = log_dataset_no_duplicates(
        #     session_train, feature_type=params["feature_type"], data_pct=1, shuffle=is_shuffle)
        print("len(dataset_train): ", len(dataset_train))

        os.makedirs(params["wtp_records_path"], exist_ok=True)
        if params["is_validation"]:
            if params["use_val_template_in_learning"]:
                ext.update_id2log_train(session_val)
            session_val = ext.transform(session_val, datatype="test")
            dataset_val = log_dataset(
                session_val, feature_type=params["feature_type"])

        if use_all_template_in_learning:
            ext.update_id2log_train(session_test)
        session_test = ext.transform(session_test, datatype="test")
        dataset_test = log_dataset(
            session_test, feature_type=params["feature_type"])

        saveDataset(name, ext, dataset_train, dataset_test, dataset_val)

    if params["is_split_dataset"]:
        if params["improved_bagging"]:
            len_split_train = int(len(dataset_train) / process_num)
            # len_split_test = int(len(dataset_test) / process_num)
            temp_dataset = copy.deepcopy(dataset_train.flatten_data_list)
            random.shuffle(temp_dataset)
            splited_train = []
            # print("aaaaaaaaaaaaaaaaaaaaaa")
            for i in range(0, len(temp_dataset), len_split_train):
                splited_train.append(temp_dataset[i: i + len_split_train])
                # print(len(temp_dataset[i: i+len_split_train]))
            if len(splited_train) != process_num:
                last = splited_train.pop(-1)
                splited_train[-1] = splited_train[-1] + last


            len_split_train = len(dataset_train)
            choise_per = 1
            choise_num = int(len_split_train*choise_per)

            for i in range(process_num):
                len_t = len(splited_train[i])
                len_t = choise_num - len_t
                splited_train[i] = splited_train[i] + random.choices(temp_dataset, k=len_t)

            print(len(splited_train))
            print(len(splited_train[0]))
            print("="*20)
        else:
            len_split_train = len(dataset_train)
            choise_per = 1
            choise_num = int(len_split_train * choise_per)
            temp_dataset = copy.deepcopy(dataset_train.flatten_data_list)
            splited_train = []
            # print("aaaaaaaaaaaaaaaaaaaaaa")
            for i in range(process_num):
                splited_train.append(random.choices(temp_dataset, k=choise_num))

            #     print(len(splited_train[-1]))
            print(len(splited_train))
            print(len(splited_train[0]))
            print("=" * 20)
        # print(temp_dataset[0])
        # sys.exit()


    # for BGL (2/3 is better for HDFS)
    if params["dataset"] == "BGL":
        params["column_dims"] = ((params["emb_dim"] // 3) * 5, (params["window_size"] // 3) * 5)
    elif params["dataset"] == "HDFS":
        params["column_dims"] = (params["emb_dim"], params["window_size"])
    params["n_templates"] = len(ext.ulog_train) + len(ext.ulog_new) + 2


    # sys.exit()



    # for pct in [0.1 * (i+1) for i in range(10)]:
    params["data_pct"] = 3
    model_save_path = dump_params(params)
    model_save_path = params["wtp_records_path"]

    store_dict = defaultdict(list)
    best_f1 = -float("inf")
    for i in range(params["n_experiment"]):
        logging.info("{}Experiment #{}{}".format("-" * 15, i, "-" * 15))
        spclfs = []
        for i in range(process_num):
            model = SPClassifierProcess(meta_data=ext.meta_data, model_save_path=model_save_path, **params)
            # model.is_learning_curve = True
            # model.wtp_records_path = params["wtp_records_path"]
            spclfs.append(model)

        if params["is_split_dataset"]:
            # src_datas = [[spclfs[num], num, splited_train[num], splited_test[num], splited_val[num]] for num in range(process_num)]
            src_datas = [[spclfs[num], num, splited_train[num], dataset_test, dataset_val] for num in range(process_num)]
        else:
            src_datas = [[spclfs[num], num, dataset_train, dataset_test, dataset_val] for num in range(process_num)]


        # if params["is_epoches"]:
        #     eval_results_epochs = []
        #     for i in range(params["epoches"]):
        #         datas = do_multiprocess(process_num, src_datas)
        #         eval_results_epochs.append(evaluate_multiprocess(datas, dataset_test))
        #     i=1
        #     for eval_results in eval_results_epochs:
        #         print("Epoch{}".format(i), eval_results)
        #         i+=1
        # else:
        #     datas = do_multiprocess(process_num, src_datas)
        #     evaluate_multiprocess(datas, dataset_test)


        if params["use_best_model"]:
            eval_results = main_best_model(ext, params, process_num, src_datas, dataset_test, dataset_val, i)
        elif params["use_best_model_with_vote"]:
            eval_results = main_best_model_with_vote(ext, params, process_num, src_datas, dataset_test, i)
        else:
            eval_results = main(params, process_num, src_datas, dataset_test, dataset_val, i)
        # sys.exit()



        if eval_results["f1"] > best_f1:
            best_f1 = eval_results["f1"]
            # model.save_model()

        for k, v in eval_results.items():
            store_dict[k].append(v)


    # model.load_model(model_save_path=model_save_path)
    whole_results = defaultdict(float)
    for k, v in store_dict.items():
        mean_score = np.mean(store_dict[k])
        whole_results[k] = mean_score

    result_str = "Final Result: " + \
                 "\t".join(["{}-{:.4f}".format(k, v) for k, v in whole_results.items()])
    print(result_str)
    print("=" * 40)

    # dump_final_results(params, whole_results)
    # dump_store_results("BGL_Store", store_dict)
    if params["is_validation"]:
        dump_store_results_cm("BGL_VAL_Store_SPCLF", store_dict)
    else:
        dump_store_results_cm("BGL_Store", store_dict)
    dump_final_results(params, whole_results)


