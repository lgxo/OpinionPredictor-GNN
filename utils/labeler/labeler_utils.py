from tqdm import tqdm
from datasets import DatasetDict, Dataset
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# 加载标签映射表
def load_labelMap(path):
    id2sentiment_dic = {}
    with open(path, "r", encoding="utf-8") as f:
        for index, emotion in enumerate(f):
            id2sentiment_dic[index] = emotion.strip()
    return id2sentiment_dic


# 加载smp数据集
def load_smp(path):
    smp_lst = []
    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            line = line.strip()
            if not line:
                continue
            text, label = line.split("\t")
            smp_lst.append({
                "text": text,
                "label": eval(label) 
            })
    return smp_lst


# 加载labeler模型
def load_labeler_Model(model_name_or_path, num_labels=None):
    if num_labels:
        model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=num_labels)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
    return model


# 加载labeler tokenizer
def load_labeler_Tokenizer(model_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    return tokenizer


# 训练模型
def train(model, tokenizer, train_samples, eval_samples, metric, **kwargs):

    train_dataset = Dataset.from_list(train_samples)
    eval_dataset = Dataset.from_list(eval_samples)

    dataset = DatasetDict({
        "train": train_dataset,
        "eval": eval_dataset  
    })

    def tokenize_function(examples):
        return tokenizer(examples["text"], **kwargs["args_tokenizer"])

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(**kwargs["args_TrainingArguments"])

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels, average="macro")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["eval"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=kwargs["early_stopping_patience"])]
    )

    trainer.train()

    pt_save_directory = kwargs["best_save_dir"]
    tokenizer.save_pretrained(pt_save_directory)
    model.save_pretrained(pt_save_directory)
    print("Best Model saved to:", pt_save_directory)


# 测试模型
def test(model, tokenizer, test_samples, label_map, **kwargs):
    test_dataset = Dataset.from_list(test_samples)
    def tokenize_function(examples):
        return tokenizer(examples["text"], **kwargs["args_tokenizer"])

    tokenized_datasets = test_dataset.map(tokenize_function, batched=True)

    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    dataloader = DataLoader(tokenized_datasets, batch_size=kwargs["test_batch_size"])

    device = kwargs["device"]
    model.to(device)

    predictions = torch.tensor([], dtype=torch.long, device=device)
    model.eval()
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.cat((predictions, torch.argmax(logits, dim=1)), dim=0)
        
    confusion_matrix_ = classification_report(tokenized_datasets["labels"].numpy(), predictions.cpu().numpy(), target_names=label_map.values(), output_dict=False)
    print(confusion_matrix_)


# 预测结果, batch_size=8, 
def predict(model, tokenizer, predict_samples, **kwargs):
    dataset = Dataset.from_list(predict_samples)
    def tokenize_function(examples):
        return tokenizer(examples["text"], **kwargs["args_tokenizer"])
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets.set_format("torch")


    dataloader = DataLoader(tokenized_datasets, batch_size=kwargs["predict_batch_size"])

    device = kwargs["device"]
    model.to(device)

    predictions = []
    model.eval()
    for batch in tqdm(dataloader, desc="Predicting"):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())

    return predictions
