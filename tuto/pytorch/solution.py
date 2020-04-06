import random
from typing import Dict, Mapping, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class ConvNeurNet(nn.Module):
    """Convolutional Neural Network for MNIST classification.

    Architecture taken form the PyTorch MNIST example.
    """

    def __init__(self) -> None:
        """Initialize the layers."""
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass of the neural network.

        Parameters
        ----------
        inputs:
            A tensor with shape [N, 28, 28] representing a set of images, where N is the
            number of examples (i.e. images), and 28*28 is the size of the images.

        Returns
        -------
        logits:
            Unnormalize last layer of the neural network (without softmax) with shape
            [N, 10], where N is the number of input example and 10 is the number of
            categories to classify (10 digits).

        """
        h = self.conv1(inputs)
        h = F.relu(h)
        h = self.conv2(h)
        h = F.relu(h)
        h = F.max_pool2d(h, 2)
        h = self.dropout1(h)
        h = torch.flatten(h, 1)
        h = self.fc1(h)
        h = F.relu(h)
        h = self.dropout2(h)
        logits = self.fc2(h)
        return logits


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    optimizer: optim.Optimizer,
    n_epoch: int,
    device: torch.device,
) -> None:
    """Train a model.

    This is the main training function that start from an untrained model and
    fully trains it.


    Parameters
    ---------
    model:
        The neural network to train.
    train_loader:
        The dataloader with the example to train on.
    valid_loader:
        The dataloder with examples used for validation.
    optimizer:
        The optimizer initialized with the model parameters.
    n_epoch:
        The number of epoch (iteration over the complete training set) to train for.
    device:
        The device (CPU/GPU) on which to train the model.

    """
    writer = SummaryWriter(flush_secs=5)
    writer.add_graph(model, next(iter(train_loader))[0])

    model.to(device)
    step = 0

    for _ in range(n_epoch):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = update_model(
                model=model, inputs=inputs, targets=targets, optimizer=optimizer
            )
            metrics = evaluate_on_batch(logits, targets)
            log_metrics(writer, metrics, step, suffix="Train")
            step += 1

            if (step % 10) == 1:
                valid_metrics = evaluate_model(model, valid_loader, device)
                log_metrics(writer, valid_metrics, step, suffix="Valid")


def update_model(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    optimizer: torch.Tensor,
) -> torch.Tensor:
    """Do a gradient descent iteration on the model.

    Parameters
    ---------
    model:
        The neural network being trained.
    inputs:
        The inputs to the model. A tensor with shape [N, 28, 28], where N is the
        number of examples.
    targets:
        The true category for each example, with shape [N].
    optimizer:
        The optimizer that applies the gradient update, initialized with the model
        parameters.

    Returns
    -------
    logits:
        Unnormalize last layer of the neural network (without softmax), with shape
        [N, 10]. Detached from the computation graph.

    """
    model.train()
    optimizer.zero_grad()
    logits = model(inputs)
    loss = F.cross_entropy(logits, targets)
    loss.backward()
    optimizer.step()
    return logits.detach()


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute the accuracy of a minibatch.

    Parameters
    ---------
    logits:
        Unnormalize last layer of the neural network (without softmax), with shape
        [N, 10], where N is the number of examples.
    targets:
        The true category for each example, with shape [N].

    Returns
    -------
    accuracy:
        As a percentage so between 0 and 100.

    """
    predictions = logits.argmax(dim=1)
    corrects = predictions == targets
    accuracy = 100 * corrects.float().sum() / corrects.size(0)
    return accuracy.item()


def evaluate_on_batch(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """Compute a number of metrics on the minibatch.

    Parameters
    ---------
    logits:
        Unnormalize last layer of the neural network (without softmax), with shape
        [N, 10], where N is the number of examples.
    targets:
        The true category for each example, with shape [N].

    Returns
    -------
    metrics:
        A dictionary mapping metric name to value.

    """
    return {
        "Loss": F.cross_entropy(logits, targets).item(),
        "Accuracy": accuracy_from_logits(logits, targets),
    }


def mean(values: Iterable[float]) -> float:
    """Compute the mean of an iterable."""
    return sum(values) / len(values)


def evaluate_model(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Dict[str, float]:
    """Compute some metrics over a dataset.

    Parameters
    ---------
    model:
        The neural network to evaluate.
    loader:
        A dataloader over a dataset. The methdo can be sued with the validation
        dataloader (during training for instance), or the testdataloder (after training
        to cpmpute the final performances).

    Returns
    -------
    metrics:
        A dictionary mapping metric name to value.

    """
    model.eval()
    metrics = {}

    with torch.no_grad():
        # Note this is not exact since the last batch may not have the
        # same number of elements, but it is a sufficient approximation
        # for our use.
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            for name, value in evaluate_on_batch(logits, targets).items():
                metrics.setdefault(name, []).append(value)
        metrics = {name: mean(values) for name, values in metrics.items()}

    return metrics


def log_metrics(
    writer: SummaryWriter, metrics: Mapping[str, float], step: int, suffix: str
) -> None:
    """Log metrics in Tensorboard.

    Parameters
    ---------
    writer:
        A the summary writer used to log the values.
    metrics:
        A dictionary mapping metric name to value.
    step:
        The value on the abscissa axis for the metric curves.
    suffix:
        A string to append to the name of the metric to group them in Tensorboard.
        For instance "Train" on training data, and "Valid" on validation data.

    """
    for name, value in metrics.items():
        writer.add_scalar(f"{name}/{suffix}", value, step)
