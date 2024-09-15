import os


class Baseline:
    def __init__(self, model_name_or_path, checkpoints_dir, reference_urls, **kwargs):
        self.__model_name_or_path = model_name_or_path
        self.__checkpoints_dir = checkpoints_dir
        self.__reference_urls = reference_urls
        self.__config = None

        self.__load_config()
        self.__update_config(**kwargs)
        
    def __load_config(self):
        self.__config = {
            "baseline_dir": "data/baseline",    # where the dataset is stored
            # "min-X": 4,
            # "channel": channel,

            "setup_seed": 2024,

            "num_labels": 6,

            # "model_name": baseline_name,
            # "device": device_setting(),
            "device": "cuda",

            # "best_save_dir": best_save_dir,
            "best_save_dir": None,

            # tokenizer
            "args_tokenizer":{
                "truncation":True,
                "padding":"max_length",
                # "padding":"longest",
                "return_tensors": "pt",
            },

            # TrainingArguments
            "args_TrainingArguments":{
                # "output_dir": output_dir,
                "output_dir": None,
                "overwrite_output_dir": False,

                "seed": 2024,

                "num_train_epochs": 50,
                "per_device_train_batch_size": 16,
                "per_device_eval_batch_size": 16,

                "save_strategy": "steps",
                "save_steps": 100,

                "evaluation_strategy": "steps",
                "eval_steps": 100,

                "load_best_model_at_end": True,
                "metric_for_best_model": "KLD",
                "greater_is_better": False,
                "save_total_limit": 2,

                # "logging_dir": logs_dir,
                "logging_dir": None,
                "logging_strategy": "steps",
                "logging_steps": 100,
            },
            "metric_path": "utils/baseline/KLD.py",

            # Trainer
            "early_stopping_patience": 5,

            # test
            "test_batch_size": 8,
            # "test_result_dir": channel_dir,
            "test_result_dir": None,
        }

    def __update_config(self, **kwargs):
        for key, value in kwargs.items():
            if key=="args_TrainingArguments":
                self.__config[key].update(value)
            else:
                self.__config[key] = value

    @property
    def model_name_or_path(self):
        return self.__model_name_or_path
    
    @property
    def checkpoints_dir(self):
        return self.__checkpoints_dir
    
    @property
    def reference_urls(self):
        return self.__reference_urls
    
    def config4finetune(self, channel, device, min_x=4):
        channel_dir = f"checkpoints/baselines/{self.__checkpoints_dir}/{channel}/"
        config4finetune = self.__config.copy()
        config4finetune.update({
            "channel": channel,
            "min-X": min_x,
            "model_name": self.__model_name_or_path,
            "device": device,
            "channel_dir": channel_dir,
            "best_save_dir": os.path.join(channel_dir, "best"),
            "test_result_dir": channel_dir,
        })
        config4finetune["args_TrainingArguments"].update({
            "output_dir": os.path.join(channel_dir, "training"),
            "logging_dir": os.path.join(channel_dir, "logs"),
        })
        return config4finetune


baseline_lst = [
    # 0
    Baseline(
        model_name_or_path='lxyuan/distilbert-base-multilingual-cased-sentiments-student',
        checkpoints_dir='distilbert-base-multilingual-cased-sentiments-student',
        reference_urls = None
    ),
    # 1
    Baseline(
        model_name_or_path='HuggingFaceFW/fineweb-edu-classifier',
        checkpoints_dir='fineweb-edu-classifier',
        reference_urls = None
    ),
    # 2
    Baseline(
        model_name_or_path='distilbert/distilbert-base-uncased-finetuned-sst-2-english',
        checkpoints_dir='distilbert-base-uncased-finetuned-sst-2-english',
        reference_urls = None
    ),
    # 3
    Baseline(
        model_name_or_path='ProsusAI/finbert',
        checkpoints_dir='finbert',
        reference_urls = None
    ),
    # 4
    Baseline(
        model_name_or_path='j-hartmann/emotion-english-distilroberta-base',
        checkpoints_dir='emotion-english-distilroberta-base',
        reference_urls = None,
        # **{
        #     "args_TrainingArguments": {
        #         "per_device_eval_batch_size": 48,
        #     },
        #     "test_batch_size": 48,
        # }
    ),
    # 5
    Baseline(
        model_name_or_path='cardiffnlp/twitter-roberta-base-sentiment',
        checkpoints_dir='twitter-roberta-base-sentiment',
        reference_urls = None,
        **{
            "args_tokenizer":{
                # "padding":"max_length",
                "max_length":512,
                "padding":True,
                "truncation":True,
                "return_tensors":"pt",
            },
            "args_TrainingArguments":{
                "per_device_train_batch_size": 12,
                "per_device_eval_batch_size": 12,
            }
        }
    ),
    # 6
    Baseline(
        model_name_or_path='mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis',
        checkpoints_dir='distilroberta-finetuned-financial-news-sentiment-analysis',
        reference_urls = None,
    ),
    # 7
    Baseline(
        model_name_or_path='SamLowe/roberta-base-go_emotions',
        checkpoints_dir='roberta-base-go_emotions',
        reference_urls = None,
        **{
            "args_TrainingArguments":{
                "learning_rate": 5e-4,
                "weight_decay":0.1
            }
        }
    ),
    # 8
    Baseline(
        model_name_or_path='cardiffnlp/twitter-roberta-base-sentiment-latest',
        checkpoints_dir='twitter-roberta-base-sentiment-latest',
        reference_urls = None,
        **{
            "args_tokenizer":{
                # "padding":"max_length",
                "max_length":512,
                "padding":True,
                "truncation":True,
                "return_tensors":"pt",
            },
            "args_TrainingArguments":{
                "per_device_train_batch_size": 12,
                "per_device_eval_batch_size": 12,
                "learning_rate": 5e-4,
                "weight_decay":0.1
            }
        }
    ),
    # 9
    Baseline(
        model_name_or_path='hw2942/bert-base-chinese-finetuning-financial-news-sentiment-v2',
        checkpoints_dir='bert-base-chinese-finetuning-financial-news-sentiment-v2',
        reference_urls = None
    ),
    # 10
    Baseline(
        model_name_or_path='KernAI/stock-news-distilbert',
        checkpoints_dir='stock-news-distilbert',
        reference_urls = None,
        **{
            "args_TrainingArguments":{
                "learning_rate": 5e-4,
                "weight_decay":0.1
            }
        }
    ),
    
    # # Use `from_tf=True` to load this model from those weights.
    # Baseline(
    #     model_name_or_path='Yueh-Huan/news-category-classification-distilbert',
    #     checkpoints_dir='news-category-classification-distilbert',
    #     reference_urls = None
    # ),

    # 11
    Baseline(
        model_name_or_path='mrm8488/deberta-v3-ft-financial-news-sentiment-analysis',
        checkpoints_dir='deberta-v3-ft-financial-news-sentiment-analysis',
        reference_urls = None,
        **{
            "args_tokenizer":{
                # "padding":"max_length",
                "max_length":512,
                "padding":True,
                "truncation":True,
                "return_tensors":"pt",
            },
            "args_TrainingArguments":{
                "learning_rate": 0.0005,
                "weight_decay":0.1
            }
        }
    ),

    # # takes too long time, later
    # Baseline(
    #     model_name_or_path='fhamborg/roberta-targeted-sentiment-classification-newsarticles',
    #     checkpoints_dir='roberta-targeted-sentiment-classification-newsarticles',
    #     reference_urls = None,
    #     **{
    #         "args_tokenizer":{
    #             # "padding":"max_length",
    #             "max_length":512,
    #             "padding":True,
    #             "truncation":True,
    #             "return_tensors":"pt",
    #         },
    #         "args_TrainingArguments":{
    #             "per_device_train_batch_size": 8,
    #             "per_device_eval_batch_size": 8,
    #             "save_steps": 100,
    #             "eval_steps": 100,
    #             "logging_steps": 100,
    #         },
    #         "early_stopping_patience": 5,
    #     }
    # ),

    # 12
    Baseline(
        model_name_or_path='elozano/bert-base-cased-news-category',
        checkpoints_dir='bert-base-cased-news-category',
        reference_urls = None,
        **{
            "args_TrainingArguments":{
                "per_device_train_batch_size": 12,
                "per_device_eval_batch_size": 12,
            }
        }
    ),
    # 13
    Baseline(
        model_name_or_path='mrm8488/bert-mini-finetuned-age_news-classification',
        checkpoints_dir='bert-mini-finetuned-age_news-classification',
        reference_urls = None,
        **{
            "args_tokenizer":{
                # "padding":"max_length",
                "max_length":512,
                "padding":True,
                "truncation":True,
                "return_tensors":"pt",
            },
        }
    ),
    # 14
    Baseline(
        model_name_or_path='dima806/news-category-classifier-distilbert',
        checkpoints_dir='news-category-classifier-distilbert',
        reference_urls = None,
        **{
            "args_tokenizer":{
                # "padding":"max_length",
                "max_length":512,
                "padding":True,
                "truncation":True,
                "return_tensors":"pt",
            },
            "args_TrainingArguments":{
                "per_device_train_batch_size": 12,
                "per_device_eval_batch_size": 12,
            }
        }
    ),
    # 15
    Baseline(
        model_name_or_path='nlptown/bert-base-multilingual-uncased-sentiment',
        checkpoints_dir='bert-base-multilingual-uncased-sentiment',
        reference_urls = None,
        **{
            "args_TrainingArguments":{
                "per_device_train_batch_size": 12,
                "per_device_eval_batch_size": 12,
            }
        }
    ),
    # 16
    Baseline(
        model_name_or_path='Falconsai/intent_classification',
        checkpoints_dir='intent_classification',
        reference_urls = None
    ),
    # 17
    Baseline(
        model_name_or_path='ahmedrachid/FinancialBERT-Sentiment-Analysis',
        checkpoints_dir='FinancialBERT-Sentiment-Analysis',
        reference_urls = None,
        **{
            "args_TrainingArguments":{
                "per_device_train_batch_size": 12,
                "per_device_eval_batch_size": 12,
            }
        }
    ),
    # 18
    Baseline(
        model_name_or_path='finiteautomata/bertweet-base-sentiment-analysis',
        checkpoints_dir='bertweet-base-sentiment-analysis',
        reference_urls = None,
        **{
            "args_TrainingArguments":{
                "learning_rate": 0.01,
                "weight_decay":0.5,
                "seed": 644,
                "save_steps": 200,
                "eval_steps": 200,
                "logging_steps": 200,
            },
            "early_stopping_patience": 3,
        }
    ),
    # # 21
    # Baseline(
    #     model_name_or_path='siebert/sentiment-roberta-large-english',
    #     checkpoints_dir='sentiment-roberta-large-english',
    #     reference_urls = None
    # ),
    # # 22
    # Baseline(
    #     model_name_or_path='IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment',
    #     checkpoints_dir='Erlangshen-Roberta-110M-Sentiment',
    #     reference_urls = None
    # ),
    # # 23
    # Baseline(
    #     model_name_or_path='IDEA-CCNL/Erlangshen-Roberta-330M-Sentiment',
    #     checkpoints_dir='Erlangshen-Roberta-330M-Sentiment',
    #     reference_urls = None
    # ),
    # Baseline(
    #     model_name_or_path='',
    #     checkpoints_dir='',
    #     reference_urls = None
    # ),
]
