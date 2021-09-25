from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics.classification.accuracy import Accuracy
from transformers import AdamW, AutoModelForQuestionAnswering, AutoTokenizer

from src.metrics.jaccard import Jaccard


class TSEModel(LightningModule):
    """
    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        model: str = "roberta-base",
        tokenizer: str = "roberta-base",
        lr: float = 0.001,
        weight_decay: float = 0.0005,
        correct_bias=False,
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        self.model = AutoModelForQuestionAnswering.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()
        self.correct_bias = correct_bias
        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.val_jaccard = Jaccard()
        self.train_jaccard = Jaccard()
        self.best_val_jaccard = 0

    def forward(self, x):
        return self.model(**x)

    def step(self, batch: Any):
        outputs = self.forward(batch)
        start = self.model(**batch).start_logits.argmax(dim=1)
        end = self.model(**batch).end_logits.argmax(dim=1)
        return outputs.loss, start, end

    def convert_pred_response(self, x, start, end):
        return [
            self.tokenizer.convert_tokens_to_string(
                self.tokenizer.convert_ids_to_tokens(
                    input_id[start[index] : end[index] + 1], skip_special_tokens=True
                )
            )
            for index, input_id in enumerate(x.input_ids)
        ]

    def training_step(self, batch: Any, batch_idx: int):
        x, y = batch
        loss, start, end = self.step(x)
        preds = self.convert_pred_response(x, start, end)
        # log train metrics
        self.log("train/loss", loss)
        self.log("train/jaccard", self.train_jaccard(preds, y), on_epoch=True, on_step=True)
        # we can return here dict with any tensors
        # and then read it in some callback or in training_epoch_end() below
        # remember to always return loss from training_step, or else backpropagation will fail!
        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        x, y = batch
        loss, start, end = self.step(x)
        preds = self.convert_pred_response(x, start, end)

        # log val metrics
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        val_jaccard = self.val_jaccard(preds, y)

        self.log(
            "val/jaccard",
            val_jaccard,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        return {"loss": loss}

    def validation_epoch_end(self, outputs: List[Any]):
        val_jaccard = self.trainer.logged_metrics["val/jaccard"].cpu().item()
        if self.best_val_jaccard < val_jaccard:
            self.best_val_jaccard = val_jaccard
        self.log("val/best_jaccard", self.best_val_jaccard, on_epoch=True)

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return AdamW(
            params=self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            correct_bias=self.hparams.correct_bias,
        )
