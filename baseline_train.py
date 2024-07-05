from utils._utils import setup_seed, device_setting, read_json
from utils.baseline.baseline_utils import *
import evaluate
import os
from models.baselines import Baseline
from models.baselines import baseline_lst
import pandas as pd
import numpy as np


# finetune baseline model
def finetune(channel, baseline: Baseline, isTrain=True, isTest=True, isTrainerTest=False):
    config = baseline.config4finetune(channel, device=device_setting(), min_x=4)

    metric = evaluate.load(config["metric_path"])

    if isTrain:
        # for reproducibility
        setup_seed(config["setup_seed"])
        
        # load dataset
        train_samples = load_baseline_smp(
            os.path.join(config["baseline_dir"], f"min-{config['min-X']}", f"{config['channel']}_train.json")
        )
        eval_samples = load_baseline_smp(
            os.path.join(config["baseline_dir"], f"min-{config['min-X']}", f"{config['channel']}_dev.json")
        )

        tokenizer = load_baseline_tokenizer(config["model_name"])
        model = load_baseline_model(config["model_name"], num_labels=config["num_labels"])
        # train
        train(model, tokenizer, train_samples, eval_samples, metric, **config)

    if isTrainerTest:
        test_samples = load_baseline_smp(
            os.path.join(config["baseline_dir"], f"min-{config['min-X']}", f"{config['channel']}_test.json")
        )
        tokenizer = load_baseline_tokenizer(config["best_save_dir"])
        model = load_baseline_model(config["best_save_dir"])
        trainerTest(model, tokenizer, test_samples, metric, **config)

    if not trainerTest and isTest:
        test_samples = load_baseline_smp(
            os.path.join(config["baseline_dir"], f"min-{config['min-X']}", f"{config['channel']}_test.json")
        )
        tokenizer = load_baseline_tokenizer(config["best_save_dir"])
        model = load_baseline_model(config["best_save_dir"])

        # test
        test(model, tokenizer, test_samples, metric, **config)

def compute_acc(channel):
    checkpoint_root = "checkpoints/baselines"
    test_samples = load_baseline_smp(
        os.path.join("data/baseline", f"min-{4}", f"{channel}_test.json")
    )
    dataset = Dataset.from_list(test_samples)

    for index, baseline in enumerate(baseline_lst):
        print(f"[{index+1}/{len(baseline_lst)}]")
        print(f"Computing accuracy of {baseline.checkpoints_dir} on channel {channel}")
        best_dir = os.path.join(checkpoint_root, baseline.checkpoints_dir, channel, "best")
        config = baseline.config4finetune(channel, device=device_setting(), min_x=4)
        tokenizer = load_baseline_tokenizer(best_dir)
        model = load_baseline_model(best_dir)

        def tokenize_function(examples):
            return tokenizer(examples["text"], **config["args_tokenizer"])
        tokenized_datasets = dataset.map(tokenize_function, batched=True)

        # training_args = TrainingArguments(**config["args_TrainingArguments"])

        trainer = myTrainer(
            model=model,
            eval_dataset=tokenized_datasets,
        )

        predictions = trainer.predict(test_dataset=tokenized_datasets).predictions
        labels = np.array(tokenized_datasets["label"])
        pds = np.argmax(predictions, axis=-1)
        lbs = np.argmax(labels, axis=-1)
        acc = np.sum(pds == lbs) / len(labels)

        print(f"Accuracy of {baseline.checkpoints_dir} on channel {channel}: {acc}")
        with open(os.path.join(checkpoint_root, baseline.checkpoints_dir, channel, "acc.txt"), "w", encoding="utf-8") as f:
            f.write(str(acc))
        print("Accuracy saved to:", os.path.join(checkpoint_root, baseline.checkpoints_dir, channel, "acc.txt"))
        print()


    # for baseline_dir in os.listdir(checkpoint_root):
    #     finetune_failed = False
    #     # baseline_data = [baseline_dir]
    #     for channel in ["cj", "ty"]:
    #         best_model_path = os.path.join(checkpoint_root, baseline_dir, channel, "best")
    #         if os.path.exists(best_model_path):
    #             tokenizer = load_baseline_tokenizer(best_model_path)
    #             model = load_baseline_model(best_model_path)

    #             dataset = Dataset.from_list(test_samples)

    #             def tokenize_function(examples):
    #                 return tokenizer(examples["text"])

    #             tokenized_datasets = dataset.map(tokenize_function, batched=True)

    #             # training_args = TrainingArguments(**kwargs["args_TrainingArguments"])

    #             def compute_metrics(eval_pred):
    #                 logits, labels = eval_pred
    #                 predictions = logits
    #                 return metric.compute(predictions=predictions, references=labels)

    #             trainer = myTrainer(
    #                 model=model,
    #                 eval_dataset=tokenized_datasets,
    #                 compute_metrics=compute_metrics,
    #             )

    #             metrics_dict = trainer.evaluate()
    #             result_dict = {
    #                 "KLD": metrics_dict["eval_KLD"],
    #             }
    #             print("***** Test result *****")
    #             print(result_dict)
    #             test_result_path = os.path.join(kwargs["test_result_dir"], "test_result.txt")
    #             with open(test_result_path, "w") as f:
    #                 f.write(str(result_dict))
    #             print("Test result saved to:", test_result_path)
    #         else:
    #             finetune_failed = True
    #             break
    #     if finetune_failed:
    #         continue

def write_result2excel():
    data = []
    checkpoint_root = "checkpoints/baselines"
    for baseline_dir in tqdm(os.listdir(checkpoint_root), desc="Searching"):
        finetune_failed = False
        baseline_data = [baseline_dir]
        for channel in ["cj", "ty"]:
            test_result_path = os.path.join(checkpoint_root, baseline_dir, channel, "test_result.txt")
            if os.path.exists(test_result_path):
                with open(test_result_path, "r", encoding="utf-8") as f:
                    kv = eval(f.read().strip())
                    baseline_data.append(kv["KLD"])
            else:
                finetune_failed = True
                break

            acc_result_path = os.path.join(checkpoint_root, baseline_dir, channel, "acc.txt")
            if os.path.exists(acc_result_path):
                with open(acc_result_path, "r", encoding="utf-8") as f:
                    acc = float(f.read().strip())
                    baseline_data.append(acc)
            else:
                finetune_failed = True
                break
        if finetune_failed:
            continue
        data.append(baseline_data)
    
    df = pd.DataFrame(data=data, columns=["Model", "test_KLD(CJ)", "Acc(CJ)", "test_KLD(TY)", "Acc(TY)"])
    df.index = df.index + 1

    with pd.ExcelWriter("baseline_test_results.xlsx", mode="w") as writer:
        df.to_excel(writer, sheet_name="Baselines", index=True)   
    print("Write result to test_results.xlsx")


if __name__ == '__main__':
    # channel = "cj"
    # index_lst = [16, 17, 18, 19, 20]
    # failed_lst = []
    # for index in index_lst:
    #     # baseline = baseline_lst[index]
    #     # finetune(
    #     #     channel,
    #     #     baseline,
    #     #     isTrain=True,
    #     #     isTest=True,
    #     #     isTrainerTest=True
    #     # )
    #     try:
    #         baseline = baseline_lst[index]
    #         print(f"Training {baseline.checkpoints_dir} on channel {channel}")
    #         finetune(
    #         channel,
    #         baseline,
    #         isTrain=True,
    #         isTest=True,
    #         isTrainerTest=True
    #     )
    #     except:
    #         print(f"Failed to train {baseline.checkpoints_dir} on channel {channel}")
    #         failed_lst.append((index, baseline.checkpoints_dir))
    #         continue
    # print("\n\nSummary:")
    # for failed_index, failed_model in failed_lst:
    #     print(f"Failed to train {failed_index}:{failed_model} on channel {channel}")

    # compute_acc("ty")

    write_result2excel()


        