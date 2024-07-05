from collections import Counter
from tqdm import tqdm
import os
import pandas as pd
from ltp import LTP, StnSplit
import sys
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader


filter = __import__("utils.preprocess.filter_utils", fromlist=["CommentFilter", "TitleFilter", "NewsFilter"])
_utils = __import__("utils._utils", fromlist=["_utils"])
labeler = __import__("utils.labeler.labeler_utils", fromlist=["labeler_utils"])


# read original data
def read_original_data(path):
    original_data_lst = []

    # read original data
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        # read first data, which is not started with '[' and ended with ','
        original_data_lst.append(eval(lines[0].strip()[1:-1]))

        # read middle data, which is not ended with ','
        for line in tqdm(lines[1:-1], desc="Reading original data"):
            data = eval(line.strip()[:-1])
            original_data_lst.append(data)

        # read last data, which is not ended with ']'
        original_data_lst.append(eval(lines[-1].strip()[:-1]))

    return original_data_lst


# 重构数据，减少存储冗余。并对文本数据进行过滤。
def integrating_data(original_data_lst):
    error_data_IDs = set()
    def isErrorData(data):
        if data['news_id'] in error_data_IDs:
            return True
    
    new_data_dict = dict()   # {news_id: {news_channel: "", news_title: "", news_content: "", comment_ids: set(), comment_content: []}}

    if original_data_lst:
        print("Original data read successfully.")
        for original_data in tqdm(original_data_lst, desc="Integrating data"):
            if isErrorData(original_data):
                continue

            # pop unnecessary fields
            """
            original fields 13: ['news_id', 'news_time', 'news_channel', 'news_media', 'news_title', 'news_tags', 'comment_num', 'news_url', 'news_content', 'comment_id', 'comment_agree', 'comment_content', 'comment_lens']
            uselfull fields 6: ['news_id', 'news_channel', 'news_media', 'news_title', 'news_content', 'comment_id', 'comment_content']
            useless fields 7: ['news_time', 'news_tags', 'comment_num', 'news_url', 'comment_agree', 'comment_lens']
            """
            useless_fields = [
                'news_time', 
                'news_tags', 
                'comment_num', 
                'news_url', 
                'comment_agree', 
                'comment_lens'
            ]
            for field in useless_fields:
                original_data.pop(field)

            # integrate data
            value_dict = new_data_dict.setdefault(original_data['news_id'], dict())
            # if news_id already exists, update the value_dict
            if value_dict:
                # if comment id not in comment_ids, add it to the set
                if not original_data["comment_id"] in value_dict["comment_ids"]:
                    filtered_comment = filter.CommentFilter(original_data["comment_content"])
                    if not filtered_comment:
                        continue
                    else:
                        value_dict["comment_ids"].add(original_data["comment_id"])
                        value_dict["comment_text"].append(filtered_comment)
            # if news_id not exists, create a new value_dict, and add the original data to it
            else:
                """
                useful_fields = [
                'news_id', # not be used here
                'news_channel', 
                'news_media',
                'news_title', # No need to record
                'news_content', 
                'comment_id', 
                'comment_content'
                ]
                """
                value_dict['channel'] = original_data['news_channel']

                # 没有title，没必要存储
                filtered_title = filter.TitleFilter(original_data['news_title'])
                if not filtered_title:
                    new_data_dict.pop(original_data['news_id'])
                    error_data_IDs.add(original_data['news_id'])
                    continue
                else:
                    value_dict['title'] = filtered_title

                # 没有news，没必要存储
                filtered_news = filter.NewsFilter(original_data['news_content'], original_data['news_media'])
                if not filtered_news:
                    new_data_dict.pop(original_data['news_id'])
                    error_data_IDs.add(original_data['news_id'])
                    continue
                else:
                    value_dict['news_text'] = filtered_news

                # 没有评论，跳过该条评论
                filtered_comment = filter.CommentFilter(original_data['comment_content'])
                if not filtered_comment:
                    value_dict["comment_ids"] = set()
                    continue
                else:
                    value_dict['comment_ids'] = set(original_data["comment_id"])
                    value_dict['comment_text'] = [filtered_comment]

    return new_data_dict


# Divide data by channel, and save them into different files. 
# Bisides, preliminarily drop out the data without enough comment number(threshold).
def channel_division(data_dict, threshold=0):
    channel_division_dict = dict()
    # divide data by channel
    for news_id, value_dict in data_dict.items():
        channel = value_dict.pop("channel")
        channel_news = channel_division_dict.setdefault(channel, list())

        # drop out the data without enough comment number
        if len(value_dict["comment_text"]) < threshold:
            continue
        value_dict.pop("comment_ids")
        value_dict["news_id"] = news_id
        channel_news.append(value_dict)
    return channel_division_dict


# Save data by channel
def save_data(channel_division_dict, save_dir):
    # save data by channel
    for channel, channel_news in channel_division_dict.items():
        # with open(os.path.join(save_dir, f"{channel}.json"), "w", encoding="utf-8") as f:
        #     _utils.write_json(channel_news, f)
        _utils.write_json(channel_news, os.path.join(save_dir, f"{channel}.json"))
        print(f"{channel} data saved successfully.")


# Preview the number of news in each channel
def preview(dir):
    data = []
    channel_file_lst = os.listdir(dir)
    sentence_Spliter = load_split_model()
    for channel_file in channel_file_lst:
        # Skip non-json files
        if not channel_file.endswith(".json"):
            continue
        file_path = os.path.join(dir, channel_file)

        # with open(file_path, "r", encoding="utf-8") as f:
        #     js_lst = _utils.read_json(f)
        js_lst = _utils.read_json(file_path)

        title_len_lst = []
        news_sent_len_lst = []
        content_len_lst = []
        comment_len_lst = []
        for js in js_lst:
            title_len_lst.append(len(js["title"]))
            content_len_lst.append(len(js["title"])+len(js["news_text"]))
            comment_len_lst.append([len(comment) for comment in js["comment_text"]])
            news_sent_len_lst.append([len(sentence) for sentence in sentencesSpliter(sentence_Spliter, js["news_text"])])

        channel = channel_file[:-5]

        news_num = len(js_lst)
        total_title_len = sum(title_len_lst)
        total_comment_num = sum([len(lst) for lst in comment_len_lst])
        total_content_len = sum(content_len_lst)
        total_comment_len = sum([sum(lst) for lst in comment_len_lst])

        total_news_sent_num = sum([len(lst) for lst in news_sent_len_lst])
        total_news_sent_len = sum([sum(lst) for lst in news_sent_len_lst])

        loc_data = {}
        loc_data["channel"] = channel
        loc_data["news_num"] = news_num
        loc_data["avg_comment_num"] = total_comment_num/news_num
        loc_data["avg_content_len"] = total_content_len/news_num
        loc_data["avg_comment_len"] = total_comment_len/total_comment_num
        loc_data["avg_title_len"] = total_title_len/news_num
        # without title
        loc_data["avg_news_sent_num"] = total_news_sent_num/news_num
        loc_data["avg_news_sent_len"] = total_news_sent_len/total_news_sent_num
        loc_data["total_comment_num"] = total_comment_num

        data.append(loc_data)

    # Sort by news number in descending order
    data.sort(key=lambda e:e["news_num"], reverse=True)

    # Save the data to a csv file
    columns = ["channel", "news_num", "avg_comment_num", "avg_content_len", "avg_comment_len", "avg_title_len", "avg_news_sent_num", "avg_news_sent_len", "total_comment_num"]
    df = pd.DataFrame(data=data, columns=columns)
    df.to_csv(os.path.join(dir, "num_preview.csv"))
    print("num_preview.csv file generation completed.")
    # display the statistics
    print(df)


# load segment model
def load_split_model():
    splitModel = StnSplit()
    return splitModel


# split news content into sentences, and segment each sentence.
def sentencesSpliter(splitModel, paragraph_text):
    return splitModel.split(paragraph_text)


# load segment model
def load_segment_model(model_name_or_path, cache_dir, local_files_only, device):
    segmentModel = LTP(
        pretrained_model_name_or_path = model_name_or_path,
        cache_dir = cache_dir,
        local_files_only = local_files_only,
    )
    # move model to device
    segmentModel.to(device)
    return segmentModel
    # pipe = pipeline(model=cache_dir)
    # return pipe


# segment
def sentenceSegmenter(segmentModel, samples_lst, batch_size):
    # return segmentModel.pipeline(sentence_lst, tasks = ["cws"], return_dict = True).get("cws")
    dataset = Dataset.from_list(samples_lst)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    results = []
    for batch in dataloader:
        outputs = segmentModel.pipeline(batch["text"], tasks = ["cws"], return_dict = True).get("cws")
        results.extend(outputs)
    return results


# generate id to prediction map
def id2predictions_map(news_id_lst, prediction_lst):
    _map = dict()
    for id, prediction in zip(news_id_lst, prediction_lst):
        predictions = _map.setdefault(id, [])
        predictions.append(prediction)
    return _map
    # for js in tqdm(js_lst, desc="Matching data"):
    #     js["label"] = _map.get(js["news_id"])


# Segment and drop out stopwords for news title and content using ltp.
# Besides,match the sentiment distribution of news comments using sentiment analysis model.
def handle(
        segmentModel,
        splitModel,
        predictModel,
        predictTokenizer,
        js_lst,
        stopwords_set,
        **kwargs
):
    news_id_lst = []
    comment_sample_lst = []

    print("Start processing data...")
    startTime = _utils.get_StartTime()
    for js in tqdm(js_lst[:], desc="Processing data"):
        # segment news title and content
        title_text = js.pop("title")
        title_sample_lst = [{"text": title_text}]
        seg_title = sentenceSegmenter(segmentModel, title_sample_lst, batch_size=kwargs["segment_batch_size"])
        seg_title_without_stopwords = filter.stopwordsFilter(seg_title[0], stopwords_set)
        if not seg_title_without_stopwords:
            js_lst.remove(js)
            continue
        else:
            js["seg_title"] = [seg_title_without_stopwords]

        content_text = js.pop("news_text")
        sentence_sample_lst = [{"text": sentence} for sentence in sentencesSpliter(splitModel, content_text)] # may have blank sentence
        # js["seg_content"] = [filter.stopwordsFilter(seg_sentence, stopwords_set) for seg_sentence in sentenceSegmenter(segmentModel, sentence_lst)]
        seg_content = []
        for seg_sentence in sentenceSegmenter(segmentModel, sentence_sample_lst, batch_size=kwargs["segment_batch_size"]):
            filtered_seg_sentence = filter.stopwordsFilter(seg_sentence, stopwords_set)
            if filtered_seg_sentence:   # handle blank sentence
                seg_content.append(filtered_seg_sentence)
        if not seg_content:
            js_lst.remove(js)
            continue
        else:
            js["seg_content"] = seg_content

        """
        # Use sentiment analysis model process data togather and then mathch the result is more efficient.
        # predict sentiment distribution of news comments
        # print(f"Processing {index+1}/{news_num}... ", end="\n")
        comment_lst = js.pop("comment_text")
        # sample_lst = [{"text": comment} for comment in comment_lst]
        # prediction = labeler.predict(predictModel, predictTokenizer, sample_lst, **kwargs)
        # counter = Counter(prediction)
        # # 6 is the number of sentiment categories
        # distribution = [counter.get(key, 0) for key in range(6)]
        # js["label"] = distribution
        """
        # process togather
        comment_lst = js.pop("comment_text")
        partial_sample_lst = [{"text": comment} for comment in comment_lst]
        comment_sample_lst.extend(partial_sample_lst)
        news_id_lst.extend(js["news_id"] for _ in range(len(partial_sample_lst)))
    prediction_lst = labeler.predict(predictModel, predictTokenizer, comment_sample_lst, **kwargs)

    # match the sentiment distribution of news comments with the news data
    _map = id2predictions_map(news_id_lst, prediction_lst)

    for js in tqdm(js_lst, desc="Matching data"):
        distribution = _map.get(js["news_id"])
        counter = Counter(distribution)
        # 6 is the number of sentiment categories
        distribution = [counter.get(key, 0) for key in range(6)]
        js["label"] = distribution

    print(f"Handling timeused: {_utils.get_timeUsed(startTime)}s")
    return js_lst
