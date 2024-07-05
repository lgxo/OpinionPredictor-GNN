import os
from utils._utils import get_StartTime, get_timeUsed, read_json, write_json
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
from utils.dataset.dataset_utils import *


# generate data for baseline and myMethod
def data_generator(channel, **kwargs):
    # load data
    print("Loading data ...")
    data_myMethod = read_json(os.path.join(kwargs["save_dir"], channel + ".json"))
    data_baseline = read_json(os.path.join(kwargs["divided_dir"], channel + ".json"))
    print("Loading data done.")

    # default filter threshold
    default_minimum_limit = 3
    filter_threshold = max(default_minimum_limit, kwargs["categories_interval"][0])
    print("Filtering data ...")
    filter_data(data_myMethod, data_baseline, filter_threshold)
    print("Filtering data done.")

    print("Splitting data into train, dev, test sets...")
    dict_myMethod, dict_baseline = split(
        data_myMethod,
        data_baseline,
        seeds=kwargs["split_seed"],
        dev_size=kwargs["dev_size"],
        test_size=kwargs["test_size"],
    )
    print("Splitting data done.")

    # save data for baseline
    # print("Saving data for baseline...")
    # for _type, data in dict_baseline.items():
    #     write_json(data, os.path.join(kwargs["baseline_dir"], f"min-{filter_threshold}", channel + "_" + _type + ".json"))
    # print("Saving data for baseline done.")

    # save data for myMethod
    w2v_model = load_w2v_model(kwargs["w2v_path"], kwargs["w2v_bz2_path"])
    w2v_corpus = load_w2v_corpus(w2v_model)
    print("Saving data for myMethod...")
    for _type, data in dict_myMethod.items():
        # 为data进行聚类，得到中心向量，指定top-k
        # cluster(data, top_k, w2v_model, w2v_corpus)
        # 利用标题和中心向量，生成图格式数据集（标签从列表转换成pt）
        dataset4myMethod, dataset4myMethod_offset_lst = dataset_constructer(
            data=data, 
            interval=kwargs["categories_interval"],
            top_k=kwargs["top_k"],
            similarity_threshold=kwargs["similarity_threshold"],
            random_state = kwargs["kmeans_random_state"],
            w2v_model=w2v_model,
            w2v_corpus=w2v_corpus,
            # offset_lst = [-2, -1, 0, 1, 2],
            offset_lst=kwargs["offset_lst"],
        )
        root_dir = os.path.join(
            kwargs["data4myMethod_dir"],
            f"interval({kwargs['categories_interval'][0]}-{kwargs['categories_interval'][1]})",
            f"top{kwargs['top_k']}-{kwargs['similarity_threshold']}"
            # str(kwargs["similarity_threshold"]),
        )
        # 保存该数据集{}
        Dataset4MyMethod(
            root=root_dir,
            channel=channel,
            _type=_type,
            dataset=dataset4myMethod
        )
        for index, offset in enumerate(kwargs["offset_lst"]):
            offset_dir = os.path.join(root_dir, f"k{offset:+}")
            Dataset4MyMethod(
                root=offset_dir,
                channel=channel,
                _type=_type,
                dataset=dataset4myMethod_offset_lst[index]
            )


if __name__ == '__main__':
    # generator_config = {
    #     "divided_dir": "data/corpus/divided",
    #     "save_dir": "data/corpus/preprocessed",
    #     "baseline_dir": "data/baseline",
    #     "data4myMethod_dir": "data/data4myMethod",
    #     "w2v_path": "data/resources/w2v/w2v.kv",
    #     "w2v_bz2_path": "data/resources/w2v/sgns.weibo.word.bz2",
        
    #     "categories_interval": (4, 15),
    #     "top_k": 1, # start from 1
    #     "similarity_threshold": 0.5,
    #     "kmeans_random_state": 42,
    #     "offset_lst": [-2, -1, 0, 1, 2],

    #     "split_seed": [644, 644],
    #     "dev_size": 0.2,
    #     "test_size": 0.2,
    # }
    # test_channel = ["cj", "ty"]
    # # test_channel = ["yl"]
    # for channel in test_channel:
    #     print("Generating data for channel: ", channel)
    #     data_generator(channel, **generator_config)
    #     print("Generating data for channel: ", channel, "done.\n")

    # preview size of corpus: 195202
    w2v_model = load_w2v_model("data/resources/w2v/w2v.kv", "data/resources/w2v/sgns.weibo.word.bz2")
    w2v_corpus = load_w2v_corpus(w2v_model)
