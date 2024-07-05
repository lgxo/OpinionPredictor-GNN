"""Kullback–Leibler divergence metric."""

import datasets
import torch
import torch.nn.functional as F

import evaluate


_DESCRIPTION = """
to do
"""


_KWARGS_DESCRIPTION = """
to do
"""


_CITATION = """
to do
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class KLD(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("float32")),
                    "references": datasets.Sequence(datasets.Value("float32")),
                }
                # if self.config_name == "multilabel"
                # else {
                #     "predictions": datasets.Value("float32"),
                #     "references": datasets.Value("float32"),
                # }
            ),
            # 可参考该链接，撰写注释
            reference_urls=["https://huggingface.co/spaces/evaluate-metric/accuracy/blob/main/accuracy.py"],
        )

    def _compute(self, predictions, references, average=None, sample_weight=None):
        with torch.no_grad():
            predictions = torch.tensor(predictions, dtype=torch.float32)
            references = torch.tensor(references, dtype=torch.float32)
            predictions = F.log_softmax(predictions, dim=-1)
            kl_loss = F.kl_div(predictions, references, reduction="batchmean")

        return {
            "KLD": kl_loss.item()
        }
