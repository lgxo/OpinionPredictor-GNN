import torch
from torch import nn
from torch.nn import functional as F
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from torch.utils.data import DataLoader
import os
from tqdm import tqdm


_utils = __import__('utils._utils', fromlist=["_utils"])


# load dataset for baseline
def load_baseline_smp(path):
    dataset = _utils.read_json(path)

    dataset_baseline  = []
    for data in dataset:
        # handle text field
        text = data['title']+"\n"+data['news_text']

        # handle label field
        label = F.softmax(torch.tensor(data['label'], dtype=torch.float32), dim=0)

        dataset_baseline.append({'text': text, 'label': label})
    return dataset_baseline


# 加载模型
def load_baseline_model(model_name_or_path, num_labels=None):
    if num_labels:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
    return model


# 加载tokenizer
def load_baseline_tokenizer(model_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    return tokenizer


class myTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        labels = inputs.pop("labels")

        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = nn.CrossEntropyLoss()

        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1, self.model.config.num_labels))

        return (loss, outputs) if return_outputs else loss


# 训练模型
def train(model, tokenizer, train_samples, eval_samples, metric, **kwargs):
    train_dataset = Dataset.from_list(train_samples)
    eval_dataset = Dataset.from_list(eval_samples)

    dataset = DatasetDict({
        "train": train_dataset,
        "eval": eval_dataset,
    })

    def tokenize_function(examples):
        return tokenizer(examples["text"], **kwargs["args_tokenizer"])

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(**kwargs["args_TrainingArguments"])

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits
        return metric.compute(predictions=predictions, references=labels)

    trainer = myTrainer(
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


def trainerTest(model, tokenizer, test_samples, metric, **kwargs):
    dataset = Dataset.from_list(test_samples)

    def tokenize_function(examples):
        return tokenizer(examples["text"], **kwargs["args_tokenizer"])

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # training_args = TrainingArguments(**kwargs["args_TrainingArguments"])

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits
        return metric.compute(predictions=predictions, references=labels)

    trainer = myTrainer(
        model=model,
        eval_dataset=tokenized_datasets,
        compute_metrics=compute_metrics,
    )

    metrics_dict = trainer.evaluate()
    result_dict = {
        "KLD": metrics_dict["eval_KLD"],
    }
    print("***** Test result *****")
    print(result_dict)
    test_result_path = os.path.join(kwargs["test_result_dir"], "test_result.txt")
    with open(test_result_path, "w") as f:
        f.write(str(result_dict))
    print("Test result saved to:", test_result_path)


def test(model, tokenizer, test_samples, metric,**kwargs):
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

    # predictions = torch.tensor([], dtype=torch.long, device=device)
    model.eval()
    for batch in tqdm(dataloader, desc="Testing"):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        metric.add_batch(predictions=logits, references=batch["labels"])

    result_dict = metric.compute()
    print("***** Test result *****")
    print(result_dict)
    test_result_path = os.path.join(kwargs["test_result_dir"], "test_result.txt")
    with open(test_result_path, "w") as f:
        f.write(str(result_dict))
    print("Test result saved to:", test_result_path)
