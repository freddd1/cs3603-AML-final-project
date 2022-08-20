from typing import List
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from tqdm.auto import tqdm
from sklearn.metrics import f1_score
from transformers import get_scheduler
from models import tBERT


class Trainer:
    def __init__(self,
                 model: tBERT,
                 optimizer: torch.optim,
                 train_dataloader: DataLoader,
                 num_epochs: int = 5,
                 device='cpu'):

        self.progress_bar = None
        self.device = device
        self.num_epochs = num_epochs
        self.num_training_steps = self.num_epochs * len(train_dataloader)

        self.train_dataloader = train_dataloader

        self.model = model

        self.optimizer = optimizer
        self.lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=self.num_training_steps
        )

        self.loss_func = CrossEntropyLoss()

        self.train_losses: List[List[float]] = []
        self.eval_losses: List[float] = []

    def train(self):
        self.model.train()
        self.model.to(self.device)

        self.progress_bar = tqdm(range(self.num_training_steps))
        for epoch in range(self.num_epochs):
            epoch_losses = []
            for batch in self.train_dataloader:
                batch_loss = self._train_batch(batch)
                epoch_losses.append(batch_loss)

            self.train_losses.append(epoch_losses)

    def _train_batch(self, batch: dict) -> float:
        outputs = self.model(**batch)

        labels = batch['labels'].long().to(self.device)
        loss = self.loss_func(outputs, labels)

        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()

        self.progress_bar.update(1)
        self.progress_bar.set_description(f'loss : {loss.item():.3f}')

        return loss.item()

    def evaluate(self, dataloader):
        self.model.eval()
        self.model.to(self.device)

        all_preds, all_labels = [], []

        self.progress_bar = tqdm(range(len(dataloader)))
        for batch in dataloader:
            all_labels.extend(batch['labels'])

            outputs = self.model(**batch)
            preds = torch.argmax(outputs, dim=1).tolist()
            all_preds.extend(preds)

            labels = batch['labels'].long().to(self.device)
            loss = self.loss_func(outputs, labels)

            self.eval_losses.append(loss.item())
            self.progress_bar.update(1)

        return [int(l) for l in all_labels], [int(p) for p in all_preds]
