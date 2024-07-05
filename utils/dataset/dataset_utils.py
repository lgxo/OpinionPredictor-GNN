from sklearn.model_selection import train_test_split
from typing import List, Union, Tuple
import os
import numpy as np
from gensim.models import KeyedVectors
from utils._utils import get_StartTime, get_timeUsed
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import heapq
import torch
from torch_geometric.data import Dataset, Data
from tqdm import tqdm
import warnings
from sklearn.exceptions import ConvergenceWarning
# 关闭k-means的ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# filter data based on the length of the number of sentences in the news content
def filter_data(data_myMethod: List, data_baseline: List, threshold: int):
    if len(data_myMethod) != len(data_baseline):
        print("Error: arguments in filter_data() function mismatch.")
        exit()
    for index in range(len(data_myMethod)-1, -1, -1):
        if len(data_myMethod[index]["seg_content"]) < threshold:
            del data_myMethod[index]
            del data_baseline[index]
        else:
            del data_baseline[index]["comment_text"]
            data_baseline[index]["label"] = data_myMethod[index]["label"]

    # return data_baseline, data_myMethod


# split data into train, dev, and test sets
def split(
        data_myMethod: List,
        data_baseline: List,
        seeds: Union[List, Tuple],
        dev_size: float,
        test_size: float,
):
    if len(seeds) == 1:
        test_state = seeds[0]
        dev_state = seeds[0]
    else:
        test_state = seeds[0]
        dev_state = seeds[1]

    train_dev_myMethod, test_myMethod = train_test_split(data_myMethod, test_size=test_size, random_state=test_state)
    train_dev_baseline, test_baseline = train_test_split(data_baseline, test_size=test_size, random_state=test_state)

    relative_dev_size = dev_size / (1 - test_size)
    train_myMethod, dev_myMethod = train_test_split(train_dev_myMethod, test_size=relative_dev_size, random_state=dev_state)
    train_baseline, dev_baseline = train_test_split(train_dev_baseline, test_size=relative_dev_size, random_state=dev_state)

    dict_myMethod = {
        "train": train_myMethod,
        "dev": dev_myMethod,
        "test": test_myMethod
    }
    dict_baseline = {
        "train": train_baseline,
        "dev": dev_baseline,
        "test": test_baseline
    }

    return dict_myMethod, dict_baseline


# load word2vec model
def load_w2v_model(path, bz2_path=None):
    print("Loading word2vec model...")
    timeStart = get_StartTime()

    if os.path.exists(path):
        w2v_model = KeyedVectors.load(path, mmap='r')
    else:
        if bz2_path:
            w2v_model = KeyedVectors.load_word2vec_format(bz2_path, binary=False,unicode_errors='ignore')
            w2v_model.save(path)
        else:
            print("Word2vec model path required for first loading.")
            exit()

    print(f"Word2vec model loaded. Time used: {get_timeUsed(timeStart)}")

    return w2v_model


# load word2vec corpus
def load_w2v_corpus(w2v_model):
    return set(w2v_model.index_to_key)


# sentence embedding
def sentence_embedding(seg_sent, w2v_model, w2v_corpus):
    lst  = [w2v_model.get_vector(word) for word in seg_sent if word in w2v_corpus]
    if lst:
        return np.sum(lst, axis=0) / len(seg_sent)
    else:
        return np.zeros(w2v_model.vector_size)
    # return np.sum(lst, axis=0) / len(seg_sent)


# content embedding
def content_embedding(seg_content, w2v_model, w2v_corpus):
    # embedding_lst = [sentence_embedding(seg_sent, w2v_model, w2v_corpus) for seg_sent in seg_content]
    # return np.stack(embedding_lst, axis=0)
    return [sentence_embedding(seg_sent, w2v_model, w2v_corpus) for seg_sent in seg_content]


class myPriorityQueue:
    def __init__(self):
        self._heap = []
        heapq.heapify(self._heap)

    def push(self, item, prioritys: Tuple[float, int]):
        heapq.heappush(self._heap, (prioritys, item))

    def pop(self, k=1):
        nsmallest_lst = heapq.nsmallest(k, self._heap)
        return nsmallest_lst[-1][-1]
    
    @property
    def size(self):
        return len(self._heap)


# cluster content via k-means, k in categories_interval
# using top_k silhouette_score to select the best k(here k is the number of categories)
def cluster(v_lst, interval, top_k, random_state, catigories_num="auto"):
    if catigories_num == "auto":
        priorityQueue = myPriorityQueue()
        v_num = len(v_lst)
        for n_clusters in range(interval[0], min(v_num, interval[1])+1):
            kmeans = KMeans(
                n_clusters=n_clusters,
                init="k-means++",
                n_init=1,
                max_iter=300,
                random_state=random_state,
            )
            # try:
            #     cluster_labels = kmeans.fit_predict(v_lst)
            # except ValueError:
            #     print(f"Error: k-means clustering failed for n_clusters={n_clusters}.")
            #     continue
            cluster_labels = kmeans.fit_predict(v_lst)
            # 计算平均轮廓系数
            if v_num == n_clusters:
                silhouette_avg = 0
            else:
                silhouette_avg = silhouette_score(v_lst, cluster_labels)
            # silhouette_avg = silhouette_score(v_lst, cluster_labels)
            # 加入优先队列, 倾向于聚类数量较小的（还是说较大的，聚类数多特征区分度越高？得到的信息越多？）
            # 轮廓系数越大越好，因此取负值
            priorityQueue.push(kmeans, (-silhouette_avg, n_clusters))

        # select the top-k using silhouette_score
        best_kmeans = priorityQueue.pop(top_k)

    # specify the number of categories
    else:
        if isinstance(catigories_num, int):
            best_kmeans = KMeans(
                n_clusters=catigories_num,
                init="k-means++",
                n_init=1,
                max_iter=300,
                random_state=random_state,
            )
            best_kmeans.fit(v_lst)
        else:
            print("Error: catigories_num should be an integer.")
            exit()
    
    # return the cluster centers
    return best_kmeans.cluster_centers_
    

# compuate similarity between two vectors
# return: similarity score
def Similarity(v1, v2):
    similarity = cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))[0][0]
    # cosine_similarity([v1], [v2])
    return similarity


# def isInInterval(value, interval, delta):

def edge_constructer(nodes_v, similarity_threshold):
    edge_lst = [[], []]
    edge_weight_lst = []
    # Calculate edges
    for i in range(len(nodes_v)):
        for j in range(i+1, len(nodes_v)):
            similarity = Similarity(nodes_v[i], nodes_v[j])
            # similarity less than threshold, skip
            if similarity < similarity_threshold :
                continue
            # add edge to edge_index_tuple_lst
            else:
                edge_lst[0].extend([i, j])
                edge_weight_lst.append([similarity])

                edge_lst[1].extend([j, i])
                edge_weight_lst.append([similarity])
    return edge_lst, edge_weight_lst


# offset_lst = None, 只返回完整数据集
# offset_lst != [], 返回的第一个是完整数据集，剩余的为过滤后且k经过偏置的数据集，此时需要指定某个位置为0(未偏置，用来对比)
def dataset_constructer(data, interval, top_k, similarity_threshold, random_state, w2v_model, w2v_corpus, offset_lst=None):
    dataset_lst = []
    if offset_lst is not None:
        dataset_lst=[[] for _ in range(len(offset_lst))]
        offset_min = min(offset_lst)
        offset_max = max(offset_lst)

    dataset = []
    for datus in tqdm(data, desc="Constructing dataset", position=0):
        # print(datus["news_id"])
        # print(datus["seg_content"][8 if len(datus["seg_content"]) > 8 else 0])
        # embedding title
        title_v = sentence_embedding(datus.pop("seg_title")[0], w2v_model, w2v_corpus)

        # embedding content
        content_v_lst = content_embedding(datus.pop("seg_content"), w2v_model, w2v_corpus)
        sentence_num = len(content_v_lst)
        content_center_mt = cluster(
            v_lst=content_v_lst,
            interval=interval,
            top_k=top_k,
            random_state=random_state,
        )

        # Merge nodes
        nodes_v = np.concatenate(([title_v], content_center_mt), axis=0)

        # edge_lst = [[], []]
        # edge_weight_lst = []
        # Calculate edges
        # for i in range(len(nodes_v)):
        #     for j in range(i+1, len(nodes_v)):
        #         similarity = Similarity(nodes_v[i], content_center_mt[j])
        #         # similarity less than threshold, skip
        #         if similarity < similarity_threshold :
        #             continue
        #         # add edge to edge_index_tuple_lst
        #         else:
        #             edge_lst[0].extend([i, j])
        #             edge_weight_lst.append([similarity])

        #             edge_lst[1].extend([j, i])
        #             edge_weight_lst.append([similarity])
        # Add to dataset
        edge_lst, edge_weight_lst = edge_constructer(nodes_v, similarity_threshold)
        dataset.append({
            "nodes_v": nodes_v,
            "edges": edge_lst,
            "edge_weight": edge_weight_lst,
            "label": datus["label"],
        })

        cluster_num = len(content_center_mt)
        # 还需要在句子数内。
        cluster_inf, cluster_sup = cluster_num+offset_min, cluster_num+offset_max
        if cluster_inf > interval[0] \
            and cluster_sup < interval[1] \
            and cluster_sup < sentence_num:
            for offset, dataset_item in zip(offset_lst, dataset_lst):
                if offset == 0:
                    dataset_item.append({
                        "nodes_v": nodes_v,
                        "edges": edge_lst,
                        "edge_weight": edge_weight_lst,
                        "label": datus["label"],
                    })
                else:
                    new_content_center_mt = cluster(
                        v_lst=content_v_lst,
                        interval=interval,
                        top_k=top_k,
                        random_state=random_state,
                        catigories_num=cluster_num+offset,
                    )
                    new_nodes_v = np.concatenate(([title_v], new_content_center_mt), axis=0)
                    new_edge_lst, new_edge_weight_lst = edge_constructer(new_nodes_v, similarity_threshold)
                    dataset_item.append({
                        "nodes_v": new_nodes_v,
                        "edges": new_edge_lst,
                        "edge_weight": new_edge_weight_lst,
                        "label": datus["label"],
                    })

    return dataset, dataset_lst


class Dataset4MyMethod(Dataset):
    def __init__(self, root, channel, _type, dataset=None, device="cuda", **kwargs):
        self._type = _type
        self._channel = channel
        self._dataset = dataset
        self._device = device

        super().__init__(root)
        self._dataset = torch.load(self.processed_paths[0])
        # self.load(self.processed_paths[0])
        # if dataset:
        #     self._dataset = dataset
        # else:
        #     self._dataset = torch.load(self.processed_paths[0])
            

    @property
    def processed_file_names(self):
        processed_file_name = f'{self._channel}_{self._type}.pt'
        return [processed_file_name]

    def process(self):
        Data_lst = []
        for data in tqdm(self._dataset, desc="Transforming dataset"):
            # debug shape of following tensors
            # shape [num_nodes, num_node_features]
            x = torch.tensor(data["nodes_v"], dtype=torch.float32)
            # shape [2, num_edges]
            edge_index = torch.tensor(data["edges"], dtype=torch.long)
            # shape [num_edges, num_edge_features]
            edge_attr = torch.tensor(data["edge_weight"], dtype=torch.float32)
            # shape [1, *]
            label_without_softmax = torch.tensor(data["label"], dtype=torch.float32)
            label = torch.nn.functional.softmax(label_without_softmax.reshape(1, -1), dim=1)

            graph_Data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                labels=label,
            ).to(self._device)
            Data_lst.append(graph_Data)

        # self.save(Data_lst, self.processed_paths[0])
        torch.save(Data_lst, self.processed_paths[0])

    def len(self):
        return len(self._dataset)

    def get(self, idx):
        return self._dataset[idx]
