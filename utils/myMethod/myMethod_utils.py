import heapq
import os
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.dataset.dataset_utils import Dataset4MyMethod
from torch.optim.lr_scheduler import LinearLR, ExponentialLR


models_gcn = __import__("models.myGraph_gcn", fromlist=["myGraph_gcn"])
models_gat = __import__("models.myGraph_gat", fromlist=["myGraph_gat"])


class myQueue:
    def __init__(self, length):
        self.length = length
        self.heap = []

    def put(self, iter_count, metric_value):
        _iter_count = None
        heapq.heappush(self.heap, (-metric_value, iter_count))
        if len(self.heap) > self.length:
            _priority, _iter_count = heapq.heappop(self.heap)
        return _iter_count
    
    def get_best(self):
        _priority, _item_count = heapq.nsmallest(self.length, self.heap)[-1]
        return _item_count
    

class MyArguments:
    def __init__(self, **kwargs) -> None:
        self.output_dir = kwargs["output_dir"]
        self.train_batch_size = kwargs["train_batch_size"]
        self.eval_batch_size = kwargs["eval_batch_size"]
        self.learning_rate = kwargs["learning_rate"]
        self.weight_decay = kwargs["weight_decay"]
        self.num_train_epochs = kwargs["num_train_epochs"]
        self.eval_steps = kwargs["eval_steps"]
        self.log_dir = kwargs["log_dir"]
        self.log_steps = kwargs["log_steps"]
        self.save_total_limit = kwargs["save_total_limit"]
        self.metric_for_best_model = kwargs["metric_for_best_model"]
        self.early_stopping_patience = kwargs["early_stopping_patience"]


class MyTrainer:
    def __init__(self, model, args: MyArguments, train_dataset, eval_dataset, metric):
        self.args = args
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.train_dataloader = self.get_train_dataloader()
        self.eval_dataloader = self.get_eval_dataloader()

        # self.optimizer = torch.optim.Adam(
        #     self.model.parameters(),
        #     lr=self.args.learning_rate,
        #     weight_decay=self.args.weight_decay,
        # )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        self.scheduler = LinearLR(self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=self.args.num_train_epochs)
        # self.scheduler = ExponentialLR(self.optimizer, gamma=0.9)
        self.loos_fn = nn.CrossEntropyLoss(reduction="sum")
        self.last_metric = float("-inf")

        self.metric = metric
        self.summary_writer = SummaryWriter(log_dir=self.args.log_dir)

        self.step = 0
        self.total_steps = len(self.train_dataloader) * self.args.num_train_epochs

        self.best_metric = float("inf")
        self.early_stopping_counter = 0
        self.early_stopping_patience = self.args.early_stopping_patience

        self.stop_training = False
        # self.ckpt_root_dir = self.args.output_dir
        self.ckpt_dir_training = os.path.join(self.args.output_dir, "training")
        self.ckpt_dir_best = os.path.join(self.args.output_dir, "best")
        if not os.path.exists(self.ckpt_dir_training):
            os.makedirs(self.ckpt_dir_training)
        if not os.path.exists(self.ckpt_dir_best):
            os.makedirs(self.ckpt_dir_best)

        self.priorityQueue = myQueue(self.args.save_total_limit)

    def get_train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.args.train_batch_size, shuffle=True)

    def get_eval_dataloader(self, eval_dataset=None):
        if eval_dataset is None:
            dataloader = DataLoader(self.eval_dataset, batch_size=self.args.eval_batch_size, shuffle=False)
        else:
            dataloader = DataLoader(eval_dataset, batch_size=self.args.eval_batch_size, shuffle=False)
        return dataloader

    def training_step(self, inputs):
        self.model.train()
        self.optimizer.zero_grad()
        labels = inputs.pop("labels")
        outputs = self.model(**inputs)
        logits = outputs.get("logits")
        loss = self.loos_fn(logits, labels)
        loss.backward()
        self.optimizer.step()
        # self.scheduler.step()
        metric_dict = self.metric.compute(predictions=logits, references=labels)
        metric_dict["train_loss"] = loss.item()/len(labels)
        metric_dict["train_KLD"] = metric_dict.pop("KLD")
        return metric_dict

    def evaluate_step(self, inputs):
        with torch.no_grad():
            labels = inputs.pop("labels")
            outputs = self.model(**inputs)
            logits = outputs.get("logits")
            loss = self.loos_fn(logits, labels)
            self.metric.add_batch(predictions=logits, references=labels)
        return loss.item()

    def train(self):
        with tqdm(total=self.total_steps, position=0) as pbar:
            for _ in range(self.args.num_train_epochs):
                for batch in self.train_dataloader:
                    self.step += 1
                    pbar.update(1)
                    metric_dict = self.training_step(batch)
                    if self.step % self.args.eval_steps == 0:
                        eval_dict = self.evaluate()
                        for key, value in eval_dict.items():
                            self.summary_writer.add_scalar(f"{key}", value, self.step)
                        print(eval_dict)
                        self.save_checkpoint(eval_dict)
                        self.early_stopping()
                        if self.stop_training:
                            break
                    if self.step % self.args.log_steps == 0:
                        metric_dict["train_epoches"] = self.step/len(self.train_dataloader)
                        for key, value in metric_dict.items():
                            self.summary_writer.add_scalar(f"{key}", value, self.step)
                        # print(metric_dict)

                # self.scheduler.step()
                if self.stop_training:
                    break
        print(metric_dict)
        # 训练结束加载最优模型
        best_step = self.priorityQueue.get_best()
        ckpt_path = os.path.join(self.ckpt_dir_training, f"checkpoint-{best_step}.ckpt")
        self.model.load_state_dict(torch.load(ckpt_path))
        torch.save(self.model.state_dict(), os.path.join(self.ckpt_dir_best, f"checkpoint-{best_step}.ckpt"))

    def evaluate(self, eval_dataset=None):
        self.model.eval()
        if eval_dataset is None:
            eval_dataloader = self.eval_dataloader
        else:
            eval_dataloader = self.get_eval_dataloader(eval_dataset)

        total_loss = 0
        for batch in tqdm(eval_dataloader, leave=False, position=0):
            batch_loss = self.evaluate_step(batch)
            total_loss += batch_loss
        loss = total_loss/len(eval_dataloader.dataset)

        metric_dict = self.metric.compute()
        metric_dict["eval_loss"] = loss
        metric_dict["eval_KLD"] = metric_dict.pop("KLD")
        metric_dict["epoches"] = self.step/len(self.train_dataloader)
        
        return metric_dict

    def save_checkpoint(self, metric_dict):
        step_metric = metric_dict[f"eval_{self.args.metric_for_best_model}"]
        if step_metric < self.best_metric:
            self.best_metric = step_metric
            self.early_stopping_counter = 0
            # 保存参数
            ckpt_path = os.path.join(self.ckpt_dir_training, f"checkpoint-{self.step}.ckpt")
            torch.save(self.model.state_dict(), ckpt_path)
            # 删除多余的checkpoint
            step_delete = self.priorityQueue.put(self.step, step_metric)
            if step_delete is not None:
                os.remove(os.path.join(self.ckpt_dir_training, f"checkpoint-{step_delete}.ckpt"))
        else:
            self.early_stopping_counter += 1
            # self.scheduler.step()
        if step_metric > self.last_metric:
            self.scheduler.step()
        self.last_metric = step_metric


    def early_stopping(self):
        # self.early_stopping_counter += 1
        if self.early_stopping_counter >= self.early_stopping_patience:
            print(f"Early stopping at step {self.step} because the metric is not improving")
            self.stop_training = True


def load_dataset4myMethod(_type, **kwargs):
    dir = os.path.join(
        "data/data4myMethod",
        f"interval({kwargs['interval'][0]}-{kwargs['interval'][1]})",
        f"top{kwargs['top_k']}-{kwargs['threshold']}",
    )
    offset = kwargs.get("offset", None)
    if offset is not None:
        dir = os.path.join(dir, f"k{offset:+}")
    dataset = Dataset4MyMethod(
        root=dir,
        channel=kwargs["channel"],
        _type=_type,
        dataset=None,
    )
    return dataset


# def load_model4myMethod(args, path):
#     model = None
#     model_name = args.pop("model_name")
#     if model_name == "myGraph_gcn":
#         model = models_gcn.myGraph_gcn(300, 128, 6, **args).to("cuda:0")
#     elif model_name == "myGraph_gat":
#         model = models_gat.myGraph_gat(300, 128, 6, **args).to("cuda:0")
#     else:
#         raise ValueError("Invalid model name")
#     model.load_state_dict(torch.load(path))
#     return model
