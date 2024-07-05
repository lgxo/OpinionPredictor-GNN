from utils._utils import setup_seed, get_StartTime, get_timeUsed
from utils.myMethod.myMethod_utils import *
import os
from models.myGraph_gcn import myGraph_gcn
from models.myGraph_gat import myGraph_gat
import evaluate
from itertools import product
import pandas as pd
import torch.nn.functional as F


def myModel_trainer(config, random_seed=2024):
    setup_seed(random_seed)

    train_dataset = load_dataset4myMethod("train", **config)
    dev_dataset = load_dataset4myMethod("dev", **config)
    test_dataset = load_dataset4myMethod("test", **config)

    # model = myGraph_gcn(300, 128, 6).to("cuda:0")
    # # model = myGraph_gat(300, 128, 6).to("cuda:0")

    output_dir = os.path.join(
        config["checkpoint_root_dir"],
        model.__class__.__name__,
        config["channel"],
        f"interval({config['interval'][0]}-{config['interval'][1]})",
        f"top{config['top_k']}-{config['threshold']}"
    )
    offset = config.get("offset", None)
    if offset is not None:
        output_dir = os.path.join(output_dir, f"k{offset:+}")
    log_dir = os.path.join(output_dir, "logs")

    myArgs = MyArguments(
        output_dir=output_dir,

        learning_rate=0.005,
        num_train_epochs=500,
        # weight_decay = 1e-4,
        weight_decay = 0,

        train_batch_size=16,
        eval_batch_size=16,
        
        eval_steps=100,

        log_dir=log_dir,
        log_steps=100,

        save_total_limit=2,
        metric_for_best_model="KLD",
        early_stopping_patience=10,
    )
    metric = evaluate.load(config["metric_path"])
    myTrainer = MyTrainer(
        model = model,
        args = myArgs,
        train_dataset = train_dataset,
        eval_dataset = dev_dataset,
        metric = metric,
    )

    myTrainer.train()

    best_eval = myTrainer.evaluate(eval_dataset = dev_dataset)
    result_dict = myTrainer.evaluate(
        eval_dataset = test_dataset
    )
    # return {
    #     "eval_KLD": best_eval["eval_KLD"],
    #     "test_KLD": result_dict["eval_KLD"]
    # }
    return best_eval["eval_KLD"], result_dict["eval_KLD"]

    # print(str({"eval_KLD": best_eval["eval_KLD"]}))
    # result_dict = myTrainer.evaluate(
    #     eval_dataset = test_dataset
    # )

    # with open(os.path.join(output_dir, "test_result.txt"), "w") as f:
    #     info = str({"KLD": result_dict["eval_KLD"]})
    #     f.write(info)
    #     print(info)


def collation():
    pass


if __name__ == '__main__':
    # config = {
    #     "channel": None,
    #     "interval": (4, 15),
    #     "top_k": 1,
    #     "threshold": 0.5,
    #     "offset": None,
    #     "metric_path": "utils/baseline/KLD.py",
    #     "checkpoint_root_dir": "checkpoints/myMethod/"
    # }
    # channel_lst = ["cj", "ty"]
    # model_name_lst = ["myGraph_gcn", "myGraph_gat"]
    # layer_num_lst = [2, 3, 4]
    # readout_layer_lst = ["global_mean_pool", "global_max_pool"]
    # is_selfloops_lst = [True, False]
    # is_edge_attr_lst = [True, False]
    # # dropout_lst = [0, 0.1, 0.3, 0.5, 0.9]

    # print("="*50)
    # df_data = []
    # start_time = get_StartTime()
    # for channel, model_name, layer_num, readout_layer, is_selfloops, is_edge_attr in product(
    #     channel_lst, model_name_lst, layer_num_lst, readout_layer_lst, is_selfloops_lst, is_edge_attr_lst
    # ):
    #     dropout = 0.3
    #     print("Args:")
    #     print(f"channel: {channel}, model_name: {model_name}, layer_num: {layer_num}, readout_layer: {readout_layer}, is_selfloops: {is_selfloops}, is_edge_attr: {is_edge_attr}, dropout: {dropout}")
    #     if model_name == "myGraph_gcn":
    #         model = myGraph_gcn(300, 128, 6, layer_num, readout_layer, is_selfloops, is_edge_attr, dropout).to("cuda:0")
    #     elif model_name == "myGraph_gat":
    #         model = myGraph_gat(300, 128, 6, layer_num, readout_layer, is_selfloops, is_edge_attr, dropout).to("cuda:0")
    #     else:
    #         raise ValueError("Invalid model name")
        
    #     config["channel"] = channel
        
    #     local_data = {
    #         "channel": channel,
    #         "model_name": model_name,
    #         "layer_num": layer_num,
    #         "readout_layer": readout_layer,
    #         "is_selfloops": is_selfloops,
    #         "is_edge_attr": is_edge_attr,
    #         "dropout": dropout,
    #     }
    #     for i in range(3):
    #         eval_kld, test_kld = myModel_trainer(config, random_seed=2024)
    #         local_data[f"eval_kld_{i+1}"] = eval_kld
    #         local_data[f"test_kld_{i+1}"] = test_kld
    #     local_data["avg_eval_kld"] = sum([local_data[f"eval_kld_{i+1}"] for i in range(3)])/3
    #     local_data["avg_test_kld"] = sum([local_data[f"test_kld_{i+1}"] for i in range(3)])/3
    #     for i in range(3):
    #         print(f"eval_kld_{i+1}: {local_data[f'eval_kld_{i+1}']}, test_kld_{i+1}: {local_data[f'test_kld_{i+1}']}")
    #     print(f"avg_eval_kld: {local_data['avg_eval_kld']}, avg_test_kld: {local_data['avg_test_kld']}")
    #     df_data.append(local_data)

    #     print(f"Time used: {get_timeUsed(start_time)} s")
    #     print("\n"*3)
    #     print("="*50)

    
    # df = pd.DataFrame(df_data)
    # # df.to_csv("myMethod_result.csv", index=False)
    # with pd.ExcelWriter("myMethod_result1.xlsx", mode="w") as writer:
    #     df.to_excel(writer, sheet_name=f"drop{dropout}", index=True)   
    # print(df)
    # print("Write result to myMethod_result.xlsx")


    best_model_args = {
        "cj":{
            "model_name": "myGraph_gcn",
            "layers": 2,
            "readout_type": "global_mean_pool",
            "is_selfloop": True,
            "is_edge_attr": False,
            "dropout": 0.3,
            "best_ckpt_path": "checkpoints/myMethod/myGraph_gcn/cj/interval(4-15)/top1-0.5/best/checkpoint-2100.ckpt"
        },
        "ty":{
            "model_name": "myGraph_gcn",
            "layers": 2,
            "readout_type": "global_mean_pool",
            "is_selfloop": True,
            "is_edge_attr": True,
            "dropout": 0.1,
            "best_ckpt_path": "checkpoints/myMethod/myGraph_gcn/ty/interval(4-15)/top1-0.5/best/checkpoint-1300.ckpt"
        }
    }
    gat_model_args = {
        "cj":{
            "model_name": "myGraph_gat",
            "layers": 2,
            "readout_type": "global_mean_pool",
            "is_selfloop": True,
            "is_edge_attr": False,
            "dropout": 0.3,
            "best_ckpt_path": "checkpoints/myMethod/myGraph_gat/cj/interval(4-15)/top1-0.5/best/checkpoint-1800.ckpt"
        },
        "ty":{
            "model_name": "myGraph_gat",
            "layers": 2,
            "readout_type": "global_mean_pool",
            "is_selfloop": True,
            "is_edge_attr": True,
            "dropout": 0.3,
            "best_ckpt_path": "checkpoints/myMethod/myGraph_gat/ty/interval(4-15)/top1-0.5/best/checkpoint-1300.ckpt"
        }
    }
    channel = "cj"
    # args = best_model_args[channel]
    args = gat_model_args[channel]
    model_name = args.pop("model_name")
    best_ckpt_path = args.pop("best_ckpt_path")
    if model_name == "myGraph_gcn":
        model = myGraph_gcn(300, 128, 6, **args).to("cuda:0")
    elif model_name == "myGraph_gat":
        model = myGraph_gat(300, 128, 6, **args).to("cuda:0")
    else:
        raise ValueError("Invalid model name")
    
    config = {
        "channel": channel,
        "interval": (4, 15),
        "top_k": 1,
        "threshold": 0.5,
        "offset": None,
        "metric_path": "utils/baseline/KLD.py",
        "checkpoint_root_dir": "checkpoints/myMethod/"
    }
    # eval_kld, test_kld = myModel_trainer(config, random_seed=2024)
    # print(f"eval_kld: {eval_kld}, test_kld: {test_kld}")


    model.load_state_dict(torch.load(best_ckpt_path))
    model.to("cuda:0")

    test_dataset = load_dataset4myMethod("test", **config)
    dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    acc = 0
    kld_lst = []
    predictions_lst = []
    references_lst = []
    model.eval()
    for inputs in tqdm(dataloader):
        with torch.no_grad():
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            predictions_lst.append(F.softmax(logits, dim=-1))
            references_lst.append(labels)

            # acc = (logits.argmax(dim=-1) == labels).float().mean().item()
            # predictions = torch.tensor(predictions, dtype=torch.float32)
            # references = torch.tensor(references, dtype=torch.float32)
            predictions = F.log_softmax(logits, dim=-1)
            kl_loss = F.kl_div(predictions, labels, reduction="batchmean")
            kld_lst.append(kl_loss.item())
            if logits.argmax(dim=-1).item() == labels.argmax(dim=-1).item():
                acc += 1
    
    acc = acc/len(test_dataset)

    # top_num = 10
    # top_lst = sorted(enumerate(kld_lst), key=lambda e: e[1])[:top_num]
    print("acc:", acc)
    # print(top_lst)
    # for index, kld in top_lst:
    #     print(references_lst[index])
    #     print(predictions_lst[index])
    #     print(f"kld: {kld}")
    #     print()




        
