import click
import torch
import matplotlib.pyplot as plt
from model import ConvNet
from data import mnist
from torch import optim, nn

import seaborn as sns
sns.set_theme()

@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
@click.option("--epochs", default=20, help="number of epochs to use for training")
def train(lr, epochs):
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize the network and optimizer
    model = ConvNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_set, _ = mnist()
    loss_list = []
    model.train()
    for epoch in range(epochs):
        running_loss = 0
        for images, labels in train_set:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            output = model(images)
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()
            predicted = torch.max(output.data, 1)
            running_loss += loss.item()

        loss_list.append(loss.item())
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_set)}")

        # Save the model
        torch.save(model, f"trained_model.pt")

    # Plot a nice loss graph with epochs on the x-axis and loss on the y-axis
    plt.plot(loss_list)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    """Evaluate a trained model."""
    print("Evaluating like my life dependends on it")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_checkpoint)
    _, test_set = mnist()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_set:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on test images: {100 * correct / total}%')

cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
