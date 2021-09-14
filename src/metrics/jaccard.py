from typing import Any

import torch
from torchmetrics import Metric


class Jaccard(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("intersection", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("union", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds, target):
        current_a = [set(pred.lower().split()) for pred in preds]
        current_b = [set(t.lower().split()) for t in target]
        current_c = [a.intersection(b) for a, b in zip(current_a, current_b)]

        self.intersection += torch.sum(torch.Tensor([len(c) for c in current_c]))
        self.union += torch.sum(
            torch.Tensor(
                [len(a) + len(b) - len(c) for a, b, c in zip(current_a, current_b, current_c)]
            )
        )

    def compute(self):
        return self.intersection / self.union
