"""F1-score metric."""

import datasets
from sklearn.metrics import f1_score

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
class F1_Score(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("int32")),
                    "references": datasets.Sequence(datasets.Value("int32")),
                }
                if self.config_name == "multilabel"
                else {
                    "predictions": datasets.Value("int32"),
                    "references": datasets.Value("int32"),
                }
            ),
            # 可参考该链接，撰写注释
            reference_urls=["https://huggingface.co/spaces/evaluate-metric/accuracy/blob/main/accuracy.py"],
        )

    def _compute(self, predictions, references, average="binary", sample_weight=None):
        return {
            "f1-score": float(
                f1_score(references, predictions, average=average, sample_weight=sample_weight)
            )
        }
