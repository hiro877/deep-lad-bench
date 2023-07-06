import os
import pickle
import argparse
import sys

import pandas as pd
import numpy as np
from utils import decision, json_pretty_dump
from collections import OrderedDict, defaultdict
import random

parser = argparse.ArgumentParser()

parser.add_argument("--train_anomaly_ratio", default=1.0, type=float)

params = vars(parser.parse_args())

eval_name = f'bgl_{params["train_anomaly_ratio"]}_tar'
seed = 42
data_dir = "../data/processed/BGL/CrossVal"
np.random.seed(seed)

params = {
    "log_file": "../data/BGL/BGL.log_structured.csv",
    "time_range": 21600,  # 6 hours
    # "time_range": 1800,  # 0.5 hours
    # "time_range": 600,  # 10 minute
    # "time_range": 1,  # 10 minute
    "train_ratio": None,
    "val_ratio": 0.5,
    "test_ratio": 0.10,
    "random_sessions": False,
    "train_anomaly_ratio": params["train_anomaly_ratio"],
    "misslabeled_ratio": 0,
    "is_True2False": False,
    "is_False2True": False,
}

data_dir = os.path.join(data_dir, eval_name)
os.makedirs(data_dir, exist_ok=True)

def add_misslabel(is_True2False, is_False2True, misslabeled_ratio,
                  test_ratio, struct_log):
    if misslabeled_ratio == 0:
        return struct_log

    if is_True2False and is_False2True:
        print("MissLabelled: is_True2False and is_False2True")
        train_ratio = 1 - test_ratio
        train_labels = struct_log["Label"][:int(len(struct_log["Label"]) * train_ratio)]
        train_anomaly_indexes = [idx for idx, label in enumerate(train_labels)]
        params["train_anomaly_num_all"] = len(train_anomaly_indexes)
        train_anomaly_indexes = random.sample(train_anomaly_indexes,
                                              int(len(train_anomaly_indexes) * misslabeled_ratio))
        params["train_anomaly_num"] = len(train_anomaly_indexes) - len(train_anomaly_indexes)

        for i in train_anomaly_indexes:
            if train_labels[i] == 1:
                train_labels[i] = 0
            else:
                train_labels[i] = 1

        test_labels = struct_log["Label"][int(len(struct_log["Label"]) * train_ratio):]
        s_v = pd.concat([train_labels, test_labels])
        struct_log.drop('Label', axis=1)
        struct_log['Label'] = s_v

        # for i in train_anomaly_indexes:
        #     if struct_log["Label"][i] != 0:
        #         print("Error by Hiro. struct_log[Label][i]")
        return struct_log

    if is_True2False:
        print("MissLabelled: is_True2False")
        train_ratio = 1 - test_ratio
        train_labels = struct_log["Label"][:int(len(struct_log["Label"]) * train_ratio)]
        train_anomaly_indexes = [idx for idx, label in enumerate(train_labels) if label == 1]
        params["train_anomaly_num_all"] = len(train_anomaly_indexes)
        train_anomaly_indexes = random.sample(train_anomaly_indexes,
                                              int(len(train_anomaly_indexes) * misslabeled_ratio))
        params["train_anomaly_num"] = len(train_anomaly_indexes) - len(train_anomaly_indexes)

        for i in train_anomaly_indexes:
            if train_labels[i] != 1:
                print("Error by Hiro")
            train_labels[i] = 0

        test_labels = struct_log["Label"][int(len(struct_log["Label"]) * train_ratio):]
        s_v = pd.concat([train_labels, test_labels])
        struct_log.drop('Label', axis=1)
        struct_log['Label'] = s_v

        for i in train_anomaly_indexes:
            if struct_log["Label"][i] != 0:
                print("Error by Hiro. struct_log[Label][i]")
        return struct_log

    if is_False2True:
        print("MissLabelled: is_False2True")
        train_ratio = 1 - test_ratio
        train_labels = struct_log["Label"][:int(len(struct_log["Label"]) * train_ratio)]
        train_normal_indexes = [idx for idx, label in enumerate(train_labels) if label == 0]
        params["train_normal_num_all"] = len(train_normal_indexes)
        train_normal_indexes = random.sample(train_normal_indexes,
                                              int(len(train_normal_indexes) * misslabeled_ratio))
        # params["train_normal_num"] = len(train_labels) - len(train_normal_indexes)

        for i in train_normal_indexes:
            if train_labels[i] != 0:
                print("Error by Hiro")
            train_labels[i] = 1

        test_labels = struct_log["Label"][int(len(struct_log["Label"]) * train_ratio):]
        s_v = pd.concat([train_labels, test_labels])
        struct_log.drop('Label', axis=1)
        struct_log['Label'] = s_v

        for i in train_normal_indexes:
            if struct_log["Label"][i] != 1:
                print("Error by Hiro: After Change Labels")
        return struct_log

def add_misslabel_at_session(is_True2False, is_False2True, misslabeled_ratio, session_train):
    if misslabeled_ratio == 0:
        return session_train

    train_labels=[]
    train_session_indexes=[]
    current_idx=0
    for k, v in session_train.items():
        current_idx+=len(session_train[k]["label"])
        train_session_indexes.append(current_idx)
        train_labels.extend(session_train[k]["label"])

    if is_True2False and is_False2True:
        train_label_indexes = [idx for idx, label in enumerate(train_labels)]
        params["train_label_num_all"] = len(train_label_indexes)
        train_misslabel_indexes = random.sample(train_label_indexes,
                                              int(len(train_label_indexes) * misslabeled_ratio))
        params["train_misslabel_num"] = len(train_misslabel_indexes)

        for i in train_misslabel_indexes:
            if train_labels[i] == 1:
                train_labels[i] = 0
            else:
                train_labels[i] = 1

        current_idx=0
        for k, v in session_train.items():
            if current_idx==0:
                session_train[k]["label"] = train_labels[:train_session_indexes[current_idx]]
            else:
                session_train[k]["label"] = train_labels[train_session_indexes[current_idx-1]:train_session_indexes[current_idx]]
            current_idx+=1
        return session_train

    if is_True2False:
        train_anomaly_indexes = [idx for idx, label in enumerate(train_labels) if label == 1]
        params["train_anomaly_num_all"] = len(train_anomaly_indexes)
        train_anomaly_indexes = random.sample(train_anomaly_indexes,
                                              int(len(train_anomaly_indexes) * misslabeled_ratio))
        params["train_anomaly_num"] = len(train_anomaly_indexes)

        for i in train_anomaly_indexes:
            if train_labels[i] != 1:
                print("Error by Hiro")
            train_labels[i] = 0

        current_idx=0
        for k, v in session_train.items():
            if current_idx==0:
                session_train[k]["label"] = train_labels[:train_session_indexes[current_idx]]
            else:
                session_train[k]["label"] = train_labels[train_session_indexes[current_idx-1]:train_session_indexes[current_idx]]
            current_idx+=1
        return session_train


    if is_False2True:
        train_normal_indexes = [idx for idx, label in enumerate(train_labels) if label == 0]
        params["train_normal_num_all"] = len(train_normal_indexes)
        train_normal_indexes = random.sample(train_normal_indexes,
                                              int(len(train_normal_indexes) * misslabeled_ratio))
        params["train_normal_num"] = len(train_normal_indexes)

        for i in train_normal_indexes:
            if train_labels[i] != 0:
                print("Error by Hiro")
            train_labels[i] = 1

        current_idx=0
        for k, v in session_train.items():
            if current_idx==0:
                session_train[k]["label"] = train_labels[:train_session_indexes[current_idx]]
            else:
                session_train[k]["label"] = train_labels[train_session_indexes[current_idx-1]:train_session_indexes[current_idx]]
            current_idx+=1

        return session_train

def load_BGL(
    log_file,
    time_range,
    train_ratio,
    val_ratio,
    test_ratio,
    random_sessions,
    train_anomaly_ratio,
    misslabeled_ratio,
    is_True2False,
    is_False2True
):
    print("Loading BGL logs from {}.".format(log_file))
    struct_log = pd.read_csv(log_file, engine="c", na_filter=False, memory_map=True)
    print("ALL BGL; ", struct_log.shape)
    # struct_log.sort_values(by=["Timestamp"], inplace=True)

    struct_log["Label"] = struct_log["Label"].map(lambda x: x != "-").astype(int).values

    struct_log["time"] = pd.to_datetime(
        struct_log["Time"], format="%Y-%m-%d-%H.%M.%S.%f"
    )
    struct_log["seconds_since"] = (
        (struct_log["time"] - struct_log["time"][0]).dt.total_seconds().astype(int)
    )

    """ print anomaly templates counts """



    session_dict = OrderedDict()
    column_idx = {col: idx for idx, col in enumerate(struct_log.columns)}
    for idx, row in enumerate(struct_log.values):
        current = row[column_idx["seconds_since"]]
        if idx == 0:
            sessid = current
        # elif current - sessid > time_range:
        #     sessid = current
        if sessid not in session_dict:
            session_dict[sessid] = defaultdict(list)
        session_dict[sessid]["templates"].append(row[column_idx["EventTemplate"]])
        session_dict[sessid]["label"].append(
            row[column_idx["Label"]]
        )  # labeling for each log

    # labeling for each session
    # for k, v in session_dict.items():
    #     session_dict[k]["label"] = [int(1 in v["label"])]

    # session_idx = list(range(len(session_dict)))
    # split data
    # if random_sessions:
    #     print("Using random partition.")
    #     np.random.shuffle(session_idx)

    # session_ids = np.array(list(session_dict.keys()))

    if train_ratio is None:
        train_ratio = 1 - test_ratio
    """ Split Train, Test """
    train_lines = int(train_ratio * len(session_dict[0]["templates"]))
    test_lines = len(session_dict[0]["templates"])-train_lines
    print(train_lines)
    print(test_lines)
    print(test_lines+train_lines)
    n=0
    if n==0:
        session_train={0:
                           {"templates": session_dict[0]["templates"][0:train_lines],
                            "label": session_dict[0]["label"][0:train_lines]
                            }
                       }
        session_test={0:
                           {"templates": session_dict[0]["templates"][train_lines:],
                            "label": session_dict[0]["label"][train_lines:]
                            }
                       }
    if n>0:
        session_train={0:
                           {"templates": session_dict[0]["templates"][0:test_lines*n] + session_dict[0]["templates"][test_lines*(n+1):],
                            "label": session_dict[0]["label"][0:test_lines*n] + session_dict[0]["label"][test_lines*(n+1):]
                            }
                       }
        session_test={0:
                           {"templates": session_dict[0]["templates"][test_lines*n:test_lines*(n+1)],
                            "label": session_dict[0]["label"][test_lines*n:test_lines*(n+1)]
                            }
                       }
    print("session_train: ", len(session_train[0]["templates"]), len(session_train[0]["label"]))
    print("session_test: ", len(session_test[0]["templates"]), len(session_test[0]["label"]))

    """ Split Validation """
    val_lines = int(val_ratio * len(session_test[0]["templates"]))
    test_lines = len(session_test[0]["templates"])-val_lines
    print(test_lines)
    print(val_lines)
    print(val_lines+test_lines)
    session_dict = session_test
    if n==0:
        session_val={0:
                           {"templates": session_dict[0]["templates"][0:val_lines],
                            "label": session_dict[0]["label"][0:val_lines]
                            }
                       }
        session_test={0:
                           {"templates": session_dict[0]["templates"][val_lines:],
                            "label": session_dict[0]["label"][val_lines:]
                            }
                       }
    if n>0:
        print("nooooone")
        return
        session_val={0:
                           {"templates": session_dict[0]["templates"][0:val_lines*n] + session_dict[0]["templates"][val_lines*(n+1):],
                            "label": session_dict[0]["label"][0:val_lines*n] + session_dict[0]["label"][val_lines*(n+1):]
                            }
                       }
        session_test={0:
                           {"templates": session_dict[0]["templates"][val_lines*n:val_lines*(n+1)],
                            "label": session_dict[0]["label"][val_lines*n:val_lines*(n+1)]
                            }
                       }

    print("session_test: ", len(session_test[0]["templates"]), len(session_test[0]["label"]))
    print("session_val: ", len(session_val[0]["templates"]), len(session_val[0]["label"]))

    """ For Unsupervised """
    session_train_templates=[]
    session_train_labels=[]
    for i in range(len(session_train[0]["label"])):
        if not session_train[0]["label"][i]:
            session_train_templates.append(session_train[0]["templates"][i])
            session_train_labels.append(session_train[0]["label"][i])
    session_train[0]["templates"]=session_train_templates
    session_train[0]["label"]=session_train_labels
    print("session_train: ", len(session_train[0]["templates"]), len(session_train[0]["label"]))


    session_labels_train = sum(session_train[0]["label"])
    session_labels_val = sum(session_val[0]["label"])
    session_labels_test = sum(session_test[0]["label"])

    train_anomaly = 100 * session_labels_train / len(session_train[0]["label"])
    val_anomaly = 100 * session_labels_val / len(session_val[0]["label"])
    test_anomaly = 100 * session_labels_test / len(session_test[0]["label"])

    print("# train sessions: {} ({:.2f}%)".format(len(session_train), train_anomaly))
    print("# train sessions: {} ({:.2f}%)".format(len(session_val), val_anomaly))
    print("# test sessions: {} ({:.2f}%)".format(len(session_test), test_anomaly))
    params["session_labels_train"] = session_labels_train
    params["session_labels_val"] = session_labels_val
    params["session_labels_test"] = session_labels_test

    with open(os.path.join(data_dir, "session_train.pkl"), "wb") as fw:
        pickle.dump(session_train, fw)
    with open(os.path.join(data_dir, "session_val.pkl"), "wb") as fw:
        pickle.dump(session_val, fw)
    with open(os.path.join(data_dir, "session_test.pkl"), "wb") as fw:
        pickle.dump(session_test, fw)
    json_pretty_dump(params, os.path.join(data_dir, "data_desc.json"))
    print("Saved to {}".format(data_dir))
    return session_train, session_val, session_test


if __name__ == "__main__":
    load_BGL(**params)
