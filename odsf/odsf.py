
# [Source](https://gist.github.com/JohnnyRacer/6f53b814060d14a20c8259b0d9c9894f/revisions)

import torch
from torchmetrics import PrecisionRecallCurve

def ods_f_score(y_true, y_pred):
    """
    Calculates the Optimal Dataset Scale (ODS) F-score.
    Args:
        y_true (torch.Tensor): Ground truth labels (binary).
        y_pred (torch.Tensor): Predicted probabilities.
    Returns:
        float: ODS F-score.
    """

    precision, recall, _ = PrecisionRecallCurve(num_classes=1)(y_pred, y_true)
    f_scores = (2 * precision * recall) / (precision + recall)
    return f_scores.max().item()

# model training and prediction code here 
#
#
#
# 
# Calculating Optimal Data Scale F-measure
ods_f = ods_f_score(y_true, y_pred)
print("ODS F-score:", ods_f)


