"""Train a 2-layer MLP on MNIST and export it as an ONNX file.

The exported graph is exactly: Gemm → Relu → Gemm → Softmax
with weight names fc1.weight, fc1.bias, fc2.weight, fc2.bias,
which matches the TinyNNEngine ONNX parser expectations.
"""

import time

import click
import onnx2torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class TrainMLP(nn.Module):
    """Training model — no softmax, so CrossEntropyLoss works correctly."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ExportMLP(nn.Module):
    """Export wrapper — adds Softmax so inference engine receives probabilities."""

    def __init__(self, train_model: TrainMLP):
        super().__init__()
        self.fc1 = train_model.fc1
        self.fc2 = train_model.fc2

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x


def train(model: TrainMLP, loader: DataLoader, optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    for images, labels in loader:
        images = images.view(-1, 784).to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(images)
    return total_loss / len(loader.dataset)


def evaluate(model: TrainMLP, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.view(-1, 784).to(device)
            preds = model(images).argmax(dim=1).cpu()
            correct += (preds == labels).sum().item()
    return correct / len(loader.dataset)


@click.group()
def cli(): 
    pass



@cli.command()
@click.option("--onnx", default="mnist_fc.onnx", show_default=True,
              help="Path to the ONNX model file to evaluate.")
@click.option("--data-dir", default="data/", show_default=True,
              help="Directory for MNIST dataset (downloaded if absent).")
@click.option("--batch-size", default=256, show_default=True,
              help="Batch size for evaluation.")
@click.option("--split", default="test", type=click.Choice(["train", "test"]), show_default=True,
              help="Dataset split to evaluate on.")
def test(onnx: str, data_dir: str, batch_size: int, split: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.ToTensor()
    ds = datasets.MNIST(data_dir, train=(split == "train"), download=True, transform=transform)
    test_loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = onnx2torch.convert(onnx).to(device)
    model.eval()

    t0 = time.perf_counter()
    test_acc = evaluate(model, test_loader, device)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    n = len(ds)
    print(f"\nFinal test accuracy: {test_acc*100:.2f}%")
    print(f"End-to-end latency: {elapsed_ms:.1f} ms total  |  {elapsed_ms/n:.3f} ms/sample  ({n} samples)")
    if test_acc < 0.97:
        print("WARNING: accuracy below 97% — consider more epochs or a lower learning rate.")


@cli.command()
@click.option("--output", default="mnist_fc.onnx", show_default=True,
              help="Path to write the exported ONNX file.")
@click.option("--data-dir", default="data/", show_default=True,
              help="Directory for MNIST dataset (downloaded if absent).")
@click.option("--epochs", default=5, show_default=True,
              help="Number of training epochs.")
@click.option("--lr", default=1e-3, show_default=True,
              help="Learning rate for Adam optimizer.")
@click.option("--batch-size", default=256, show_default=True,
              help="Training batch size.")
def training(output: str, data_dir: str, epochs: int, lr: float, batch_size: int) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.ToTensor()  # normalizes pixels to [0, 1]

    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=2)

    model = TrainMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        loss = train(model, train_loader, optimizer, device)
        acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch}/{epochs}  loss={loss:.4f}  test_acc={acc*100:.2f}%")

    test_acc = evaluate(model, test_loader, device)
    print(f"\nFinal test accuracy: {test_acc*100:.2f}%")
    if test_acc < 0.97:
        print("WARNING: accuracy below 97% — consider more epochs or a lower learning rate.")

    # Build export model and run ONNX export
    export_model = ExportMLP(model).to(device).eval()
    dummy_input = torch.zeros(1, 784, device=device)

    torch.onnx.export(
        export_model,
        dummy_input,
        output,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,  # fixed batch size = 1
    )
    print(f"Saved {output}")


if __name__ == "__main__":
    cli()
