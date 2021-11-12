import math
from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import (
    AdamW,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

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
        freeze_layers=0,
        layerwise_learning_rate_decay=0,
        cosine_lr=False,
        warm_up_steps=0.
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        self.model = AutoModelForQuestionAnswering.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.freeze_layers(freeze_layers)
        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()
        self.correct_bias = correct_bias
        self.lr = lr
        self.weight_decay = weight_decay
        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.val_jaccard = Jaccard()
        self.train_jaccard = Jaccard()
        self.best_val_jaccard = 0
        self.layerwise_learning_rate_decay = layerwise_learning_rate_decay
        self.cosine_lr = cosine_lr
        self.warm_up_steps = warm_up_steps

    def forward(self, x):
        return self.model(**x)

    def freeze_layers(self, n_freeze_layers):
        if n_freeze_layers > 0:
            print(f"Freeze the first {n_freeze_layers} Layers ...")
            for param in self.model.roberta.embeddings.parameters():
                param.requires_grad = False
            for layer in self.model.roberta.encoder.layer[:n_freeze_layers]:
                for params in layer.parameters():
                    params.requires_grad = False
            print("Done.!")

    def step(self, batch: Any):
        outputs = self.forward(batch)
        start = outputs.start_logits.argmax(dim=1)
        end = outputs.end_logits.argmax(dim=1)
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
        self.log(
            "train/jaccard", self.train_jaccard(preds, y), on_epoch=True, on_step=True
        )
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
        print("Using layer-wise learning rate decay")
        params = list(self.named_parameters())
        if self.layerwise_learning_rate_decay != 0:
            grouped_parameters = self.get_optimizer_grouped_parameters(
                layerwise_learning_rate_decay=self.layerwise_learning_rate_decay,
            )
            optimizer = AdamW(
                params=grouped_parameters,
                lr=self.lr,
                weight_decay=self.weight_decay,
                correct_bias=self.correct_bias,
            )
            return optimizer
        elif self.cosine_lr:
            optimizer = AdamW(
                params=self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                correct_bias=self.correct_bias,
            )
            num_batches = len(self.train_dataloader()) * self.trainer.max_epochs
            num_warm_up_steps = (num_batches * self.warm_up_steps) / 100
            num_training_steps = num_batches - num_warm_up_steps
            return {
                "lr_scheduler": {
                    "scheduler": get_cosine_schedule_with_warmup(
                        optimizer,
                        num_warmup_steps=num_warm_up_steps,
                        num_training_steps=num_training_steps,
                    )
                },
                "optimizer": optimizer,
            }
        else:
            return AdamW(
                params=self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                correct_bias=self.correct_bias,
            )

    def get_optimizer_grouped_parameters(self, layerwise_learning_rate_decay):
        no_decay = ["bias", "LayerNorm.weight"]
        # initialize lrs for every layer
        num_layers = self.model.config.num_hidden_layers
        layers = [self.model.roberta.embeddings] + list(
            self.model.roberta.encoder.layer
        )
        layers.reverse()
        lr = self.lr
        optimizer_grouped_parameters = []
        for layer in layers:
            lr *= layerwise_learning_rate_decay
            optimizer_grouped_parameters += [
                {
                    "params": [
                        p
                        for n, p in layer.named_parameters()
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": self.weight_decay,
                    "lr": lr,
                },
                {
                    "params": [
                        p
                        for n, p in layer.named_parameters()
                        if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                    "lr": lr,
                },
            ]
        optimizer_grouped_parameters += [
            {
                "params": [
                    p
                    for n, p in self.model.qa_outputs.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.weight_decay,
                "lr": lr,
            },
            {
                "params": [
                    p
                    for n, p in self.model.qa_outputs.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
                "lr": lr,
            },
        ]
        return optimizer_grouped_parameters
