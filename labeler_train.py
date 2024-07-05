from utils._utils import setup_seed, device_setting
from utils.labeler.labeler_utils import *
import evaluate
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


config = {
    "setup_seed": 2024,

    "model_name": "google-bert/bert-base-chinese",
    "device": device_setting(),

    "class_path": "data\\smp_weibo\\class.txt",
    "train_path": "data\\smp_weibo\\train.txt",
    "eval_path": "data\\smp_weibo\\dev.txt",
    "test_path": "data\\smp_weibo\\test.txt",
    "best_save_dir": "checkpoints\\labeler\\best",

    # tokenizer
    "args_tokenizer":{
        "truncation":True,
        "padding":"max_length",
        "max_length":512,
        "return_tensors": "pt",
    },

    # TrainingArguments
    "args_TrainingArguments":{
        "output_dir": "checkpoints\\labeler\\training",
        "overwrite_output_dir": False,

        "seed": 2024,

        "num_train_epochs": 50,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 16,

        "save_strategy": "steps",
        "save_steps": 500,

        "evaluation_strategy": "steps",
        "eval_steps": 500,

        "load_best_model_at_end": True,
        "metric_for_best_model": "f1-score",
        "greater_is_better": True,
        "save_total_limit": 5,

        "logging_dir": "checkpoints\\labeler\\logs",
        "logging_strategy": "steps",
        "logging_steps": 500,
    },
    "metric_path": "utils\\labeler\\f1-score.py",

    # Trainer
    "early_stopping_patience": 10,

    # test
    "test_batch_size": 8,
}


def labeler_trainer():
    # for reproducibility
    setup_seed(config["setup_seed"])
    
    # load dataset
    label_map = load_labelMap(config["class_path"])
    tokenizer = load_labeler_Tokenizer(config["model_name"])
    model = load_labeler_Model(config["model_name"], num_labels=len(label_map))

    train_samples = load_smp(config["train_path"])
    eval_samples = load_smp(config["eval_path"])

    metric = evaluate.load(config["metric_path"])

    # train
    train(model, tokenizer, train_samples, eval_samples, metric, **config)


def labeler_test():
    label_map = load_labelMap(config["class_path"])
    tokenizer = load_labeler_Tokenizer(config["best_save_dir"])
    model = load_labeler_Model(config["best_save_dir"])

    test_samples = load_smp(config["test_path"])

    # test
    test(model, tokenizer, test_samples, label_map, **config)


if __name__ == '__main__':
    # labeler_trainer()
    labeler_test()
    