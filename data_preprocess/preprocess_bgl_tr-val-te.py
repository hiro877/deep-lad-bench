import os
import pickle
import argparse
import pandas as pd
import numpy as np
from utils import decision, json_pretty_dump
from collections import OrderedDict, defaultdict
import random

parser = argparse.ArgumentParser()

parser.add_argument("--train_anomaly_ratio", default=1.00, type=float)

params = vars(parser.parse_args())

eval_name = f'bgl_{params["train_anomaly_ratio"]}_tar'
seed = 42
data_dir = "../data/processed/BGL_VAL_RandomSessions/"
np.random.seed(seed)

params = {
    "log_file": "../data/BGL/BGL.log_structured.csv",
    "time_range": 21600,  # 6 hours
    # "time_range": 1800,  # 0.5 hours
    # "time_range": 600,  # 10 minute
    "train_ratio": 0.6,
    "val_ratio": 0.2,
    "test_ratio": 0.2,
    "random_sessions": True,
    "train_anomaly_ratio": params["train_anomaly_ratio"],
    "misslabeled_ratio": 0
}

data_dir = os.path.join(data_dir, eval_name)
os.makedirs(data_dir, exist_ok=True)


def load_BGL(
    log_file,
    time_range,
    train_ratio,
    val_ratio,
    test_ratio,
    random_sessions,
    train_anomaly_ratio,
    misslabeled_ratio
):
    print("Loading BGL logs from {}.".format(log_file))
    struct_log = pd.read_csv(log_file, engine="c", na_filter=False, memory_map=True)
    # struct_log.sort_values(by=["Timestamp"], inplace=True)

    struct_log["Label"] = struct_log["Label"].map(lambda x: x != "-").astype(int).values

    if misslabeled_ratio != 0:
        train_labels = struct_log["Label"][:int(len(struct_log["Label"])*train_ratio)]
        train_anomaly_indexes = [idx for idx, label in enumerate(train_labels) if label == 1]
        params["train_anomaly_num_all"] = len(train_anomaly_indexes)
        train_anomaly_indexes = random.sample(train_anomaly_indexes, int(len(train_anomaly_indexes)*misslabeled_ratio))
        params["train_anomaly_num"] = len(train_anomaly_indexes) - len(train_anomaly_indexes)

        for i in train_anomaly_indexes:
            if train_labels[i] != 1:
                print("Error by Hiro")
            train_labels[i] = 0
            # struct_log["Label"][i] = 0
            # struct_log.at[str(i), "Label"] = 0
            # if struct_log["Label"][i] != 0:
            #     print("Error by Hiro. struct_log[Label][i]")

        # struct_log["Label"][:int(len(struct_log["Label"])*train_ratio)] = train_labels


        test_labels = struct_log["Label"][int(len(struct_log["Label"]) * train_ratio):]
        # train_labels.extend(test_labels)
        s_v = pd.concat([train_labels, test_labels])
        struct_log.drop('Label', axis=1)
        struct_log['Label'] = s_v

        # df = pd.DataFrame({"Label", train_labels})
        # struct_log = struct_log.map({"Label": df})
        for i in train_anomaly_indexes:
            # Error Detection
            if struct_log["Label"][i] != 0:
                print("Error by Hiro. struct_log[Label][i]")


    struct_log["time"] = pd.to_datetime(
        struct_log["Time"], format="%Y-%m-%d-%H.%M.%S.%f"
    )
    struct_log["seconds_since"] = (
        (struct_log["time"] - struct_log["time"][0]).dt.total_seconds().astype(int)
    )

    session_dict = OrderedDict()
    column_idx = {col: idx for idx, col in enumerate(struct_log.columns)}
    for idx, row in enumerate(struct_log.values):
        current = row[column_idx["seconds_since"]]
        if idx == 0:
            sessid = current
        elif current - sessid > time_range:
            sessid = current
        if sessid not in session_dict:
            session_dict[sessid] = defaultdict(list)
        session_dict[sessid]["templates"].append(row[column_idx["EventTemplate"]])
        session_dict[sessid]["label"].append(
            row[column_idx["Label"]]
        )  # labeling for each log

    # labeling for each session
    # for k, v in session_dict.items():
    #     session_dict[k]["label"] = [int(1 in v["label"])]

    session_idx = list(range(len(session_dict)))
    # split data
    if random_sessions:
        print("Using random partition.")
        np.random.shuffle(session_idx)

    session_ids = np.array(list(session_dict.keys()))

    if train_ratio is None:
        train_ratio = 1 - test_ratio
    train_lines = int(train_ratio * len(session_idx))
    val_lines = int(val_ratio * len(session_idx))
    test_lines = int(test_ratio * len(session_idx))

    session_idx_train = session_idx[0:train_lines]
    session_idx_val = session_idx[train_lines: train_lines+val_lines]
    session_idx_test = session_idx[train_lines+val_lines:]

    session_id_train = session_ids[session_idx_train]
    session_id_val = session_ids[session_idx_val]
    session_id_test = session_ids[session_idx_test]
    print(session_id_val)
    print("################################")
    print(session_id_test)

    print("Total # sessions: {}".format(len(session_ids)))

    session_train = {
        k: session_dict[k]
        for k in session_id_train
        if (sum(session_dict[k]["label"]) == 0)
        or (sum(session_dict[k]["label"]) > 0 and decision(train_anomaly_ratio))
    }

    session_val = {k: session_dict[k] for k in session_id_val}

    session_test = {k: session_dict[k] for k in session_id_test}

    session_labels_train = [
        1 if sum(v["label"]) > 0 else 0 for _, v in session_train.items()
    ]
    session_labels_val = [
        1 if sum(v["label"]) > 0 else 0 for _, v in session_val.items()
    ]
    session_labels_test = [
        1 if sum(v["label"]) > 0 else 0 for _, v in session_test.items()
    ]

    train_anomaly = 100 * sum(session_labels_train) / len(session_labels_train)
    val_anomaly = 100 * sum(session_labels_val) / len(session_labels_val)
    test_anomaly = 100 * sum(session_labels_test) / len(session_labels_test)

    print("# train sessions: {} ({:.2f}%)".format(len(session_train), train_anomaly))
    print("# validation sessions: {} ({:.2f}%)".format(len(session_val), val_anomaly))
    print("# test sessions: {} ({:.2f}%)".format(len(session_test), test_anomaly))
    params["session_labels_train"] = sum(session_labels_train)
    params["session_labels_validation"] = sum(session_labels_val)
    params["session_labels_test"] = sum(session_labels_test)

    with open(os.path.join(data_dir, "session_train.pkl"), "wb") as fw:
        pickle.dump(session_train, fw)
    with open(os.path.join(data_dir, "session_val.pkl"), "wb") as fw:
        pickle.dump(session_val, fw)
    with open(os.path.join(data_dir, "session_test.pkl"), "wb") as fw:
        pickle.dump(session_test, fw)
    json_pretty_dump(params, os.path.join(data_dir, "data_desc.json"))
    print("Saved to {}".format(data_dir))
    return session_train, session_test


if __name__ == "__main__":
    load_BGL(**params)
