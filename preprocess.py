from utils._utils import read_json, write_json, device_setting
from utils.preprocess.preliminary_utils import *
from utils.labeler.labeler_utils import load_labeler_Model, load_labeler_Tokenizer, predict
from utils.preprocess.filter_utils import load_stopwords
import os


# Preliminary processing of the original data, including integrating, channel division(preliminarily drop out the data without enough comment number), and saving the data.
def preliminary(path, dir):
    original_data_lst = read_original_data(path)
    data_dict =integrating_data(original_data_lst)
    divided_dict = channel_division(data_dict, threshold=10)
    save_data(divided_dict, dir)
    print("Preliminary finished.")


# Preprocess the data, including handling, filtering, and labeling.
def preprocess(chanel_lst, **kwargs):
    # load ltp model
    print("Loading segment and split models...")
    segmentModel = load_segment_model(
        kwargs["segment_model_name_or_path"],
        kwargs["cache_dir"],
        kwargs["local_files_only"],
        kwargs["device"],
    )
    splitModel = load_split_model()
    print("Loading segment and split models finished.")

    # load sentiment prediction model
    print("Loading sentiment prediction model and tokenizer...")
    predictModel = load_labeler_Model(kwargs["predict_model_dir"])
    predictTokenizer = load_labeler_Tokenizer(kwargs["predict_model_dir"])
    print("Loading sentiment prediction model and tokenizer finished.")

    # load stopwords
    print("Loading stopwords...")
    stopwords_set = load_stopwords(kwargs["stopwords_path"])
    print("Loading stopwords finished.")

    for chanel in chanel_lst:
        print(f"***** {chanel} *****")

        # load data
        js_lst = read_json(os.path.join(kwargs["divided_dir"], chanel + ".json"))
        
        # handle data
        handled_lst = handle(
            segmentModel,
            splitModel,
            predictModel,
            predictTokenizer,
            js_lst,
            stopwords_set,
            **kwargs
        )

        # save data
        write_json(handled_lst, os.path.join(kwargs["save_dir"], chanel + ".json"))
        print(f"{chanel} handled, and new {chanel}.json saved.")
    
    print("Preprocess finished.")


if __name__ == "__main__":
    config = {
        "record_path": "data/corpus/original/record_corrected_v2.csv",
        "divided_dir": "data/corpus/divided",
        "save_dir": "data/corpus/preprocessed",

        "segment_model_name_or_path": "LTP/Base1",
        "cache_dir": "data/resources/ltp_cache/",
        "local_files_only": True,

        "predict_model_dir": "checkpoints/labeler/best",
        "stopwords_path": "data/resources/stop_words.txt",

        "device": device_setting(),

        "args_tokenizer":{
            "truncation":True,
            "padding":"max_length",
            "max_length":512,
            "return_tensors": "pt",
        },

        "segment_batch_size":256,
        "predict_batch_size":12,
    }

    # preliminary(config["record_path"], config["divided_dir"])
    preview(config["divided_dir"])

    # channel_lst = ["cj"]
    # # channel_lst = ["ty"]
    # preprocess(channel_lst, **config)
    
    # Test
    # test_channel = ["li", "yl"]
    # test_channel = ["gj"]
    # preprocess(test_channel, **config)
