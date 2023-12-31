import sys
import torch
import random
import os
import numpy as np
import h5py
import json
import pickle
import random
import hashlib
import logging
import psutil
from datetime import datetime

def check_memory(count=None):
    if (mem_use := psutil.virtual_memory().percent) > 75.0:
        logging.info(f"memory use {mem_use} on #{count}")
        if mem_use > 90:
            logging.info(f"memory use {mem_use} on #{count}: sys exit.")
            sys.exit()

def dump_final_results(params, eval_results, model):
    result_str = "\t".join(["{}-{:.4f}".format(k, v) for k, v in eval_results.items()])

    key_info = [
        "dataset",
        "train_anomaly_ratio",
        "feature_type",
        "label_type",
        "use_attention",
    ]

    args = sys.argv
    model_name = args[0].replace("_demo.py", "")
    args = args[1:]
    input_params = [
        "{}:{}".format(args[idx * 2].strip("--"), args[idx * 2 + 1])
        for idx in range(len(args) // 2)
    ]
    recorded_params = ["{}:{}".format(k, v) for k, v in params.items() if k in key_info]

    params_str = "\t".join(input_params + recorded_params)

    with open(os.path.join(f"{params['dataset']}.txt"), "a+") as fw:
        info = "{} {} {} {} {} train: {:.3f} test: {:.3f}\n".format(
            datetime.now().strftime("%Y%m%d-%H%M%S"),
            params["hash_id"],
            model_name,
            params_str,
            result_str,
            model.time_tracker["train"],
            model.time_tracker["test"],
        )
        fw.write(info)

def dump_final_results(params, eval_results):
    result_str = "\t".join(["{}-{:.4f}".format(k, v) for k, v in eval_results.items()])

    key_info = [
        "dataset",
        "train_anomaly_ratio",
        "feature_type",
        "label_type",
        "use_attention",
    ]

    args = sys.argv
    model_name = args[0].replace("_demo.py", "")
    args = args[1:]
    input_params = [
        "{}:{}".format(args[idx * 2].strip("--"), args[idx * 2 + 1])
        for idx in range(len(args) // 2)
    ]
    recorded_params = ["{}:{}".format(k, v) for k, v in params.items() if k in key_info]

    params_str = "\t".join(input_params + recorded_params)

    with open(os.path.join(f"{params['dataset']}.txt"), "a+") as fw:
        info = "{} {} {} {} {} \n".format(
            datetime.now().strftime("%Y%m%d-%H%M%S"),
            params["hash_id"],
            model_name,
            params_str,
            result_str,
        )
        fw.write(info)

def dump_store_results(save_name, store_dict):
    args = sys.argv
    model_name = args[0].replace("_demo.py", "")
    with open(os.path.join(f"{save_name}.txt"), "a+") as fw:
        info = "{} f1 recall precision acc\n".format(model_name)
        fw.write(info)
        for i in range(len(store_dict["f1"])):
            info = "{} {} {} {} {}\n".format(i,
                                           store_dict["f1"][i],
                                           store_dict["rc"][i],
                                           store_dict["pc"][i],
                                           store_dict["acc"][i],
                                           )
            fw.write(info)

def dump_store_results_cm(save_name, store_dict):
    args = sys.argv
    model_name = args[0].replace("_demo.py", "")
    with open(os.path.join(f"{save_name}.txt"), "a+") as fw:
        info = "{} f1 recall precision acc tn fp fn tp roc_auc\n".format(model_name)
        fw.write(info)
        for i in range(len(store_dict["f1"])):
            info = "{} {} {} {} {} {} {} {} {}\n".format(i,
                                           store_dict["f1"][i],
                                           store_dict["rc"][i],
                                           store_dict["pc"][i],
                                           store_dict["acc"][i],
                                                         store_dict["tn"][i],
                                                         store_dict["fp"][i],
                                                         store_dict["fn"][i],
                                                         store_dict["tp"][i],
                                                         store_dict["tp"][i],
                                                         store_dict["roc_auc"][i],
                                           )
            fw.write(info)

# def dump_store_results_cm_val(save_name, store_dict):
#     args = sys.argv
#     model_name = args[0].replace("_demo.py", "")
#     with open(os.path.join(f"{save_name}.txt"), "a+") as fw:
#         info = "{} f1 recall precision acc tn fp fn tp\n".format(model_name)
#         fw.write(info)
#         for i in range(len(store_dict["f1"])):
#             info = "{} {} {} {} {} {} {} {} {}\n".format(i,
#                                            store_dict["f1"][i],
#                                            store_dict["rc"][i],
#                                            store_dict["pc"][i],
#                                            store_dict["acc"][i],
#                                                          store_dict["tn"][i],
#                                                          store_dict["fp"][i],
#                                                          store_dict["fn"][i],
#                                                          store_dict["tp"][i],
#                                                          store_dict["tp"][i],
#                                                          store_dict["roc_auc"][i],
#                                            )
#             fw.write(info)

def dump_store_results_next_log(save_name, store_dict):
    args = sys.argv
    model_name = args[0].replace("_demo.py", "")
    with open(os.path.join(f"{save_name}.txt"), "a+") as fw:
        info = "{} f1 recall precision acc\n".format(model_name)
        fw.write(info)
        for i in range(len(store_dict["f1"])):
            info = "{} {} {} {}\n".format(i,
                                           store_dict["f1"][i],
                                           store_dict["rc"][i],
                                           store_dict["pc"][i],
                                           )
            fw.write(info)

def dump_params(params):
    hash_id = hashlib.md5(
        str(sorted([(k, v) for k, v in params.items()])).encode("utf-8")
    ).hexdigest()[0:8]
    params["hash_id"] = hash_id
    # save_dir = os.path.join("./experiment_records", hash_id)
    save_dir = os.path.join("/work2/huchida/SaveLearnedFolder/deep-lad-bench/experiment_records", hash_id)
    os.makedirs(save_dir, exist_ok=True)

    json_pretty_dump(params, os.path.join(save_dir, "params.json"))

    log_file = os.path.join(save_dir, hash_id + ".log")
    # logs will not show in the file without the two lines.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s P%(process)d %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    logging.info(json.dumps(params, indent=4))
    return save_dir


def decision(probability):
    return random.random() < probability


def json_pretty_dump(obj, filename):
    with open(filename, "w") as fw:
        json.dump(
            obj,
            fw,
            sort_keys=True,
            indent=4,
            separators=(",", ": "),
            ensure_ascii=False,
        )


def tensor2flatten_arr(tensor):
    return tensor.data.cpu().numpy().reshape(-1)


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_device(gpu=-1):
    if gpu != -1 and torch.cuda.is_available():
        device = torch.device("cuda:" + str(gpu))
    else:
        device = torch.device("cpu")
    return device


def dump_pickle(obj, file_path):
    logging.info("Dumping to {}".format(file_path))
    with open(file_path, "wb") as fw:
        pickle.dump(obj, fw)


def load_pickle(file_path):
    logging.info("Loading from {}".format(file_path))
    with open(file_path, "rb") as fr:
        return pickle.load(fr)


def save_dict_to_hdf5(dic, filename):
    """
    ....
    """
    with h5py.File(filename, "w") as h5file:
        recursively_save_dict_contents_to_group(h5file, "/", dic)


def recursively_save_dict_contents_to_group(h5file, path, dic):
    """
    ....
    """
    for key, item in dic.items():
        if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes)):
            h5file[path + key] = item
        elif isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file, path + key + "/", item)
        else:
            raise ValueError("Cannot save %s type" % type(item))


def load_dict_from_hdf5(filename):
    """
    ....
    """
    with h5py.File(filename, "r") as h5file:
        return recursively_load_dict_contents_from_group(h5file, "/")


def recursively_load_dict_contents_from_group(h5file, path):
    """
    ....
    """
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item.value
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(
                h5file, path + key + "/"
            )
    return ans
