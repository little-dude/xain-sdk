"""CNN definition for the SDK's PyTorch example"""

from typing import Any, cast

import torch
from torch import Tensor, nn, optim
from torch.nn import functional as F
from tqdm import tqdm


class Net(nn.Module):
    """[summary]

    .. todo:: Advance docstrings (https://xainag.atlassian.net/browse/XP-425)
    """

    def __init__(self) -> None:
        """[summary]

        .. todo:: Advance docstrings (https://xainag.atlassian.net/browse/XP-425)
        """

        super(Net, self).__init__()
        self.conv1: nn.Conv2d = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore  # pylint: disable=arguments-differ
        """[summary]

        .. todo:: Advance docstrings (https://xainag.atlassian.net/browse/XP-425)

        Args:
            x (~torch.Tensor): [description]

        Returns:
            ~torch.Tensor: [description]
        """

        x = cast(Tensor, self.pool(F.relu(self.conv1(x))))
        x = cast(Tensor, self.pool(F.relu(self.conv2(x))))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def train_n_epochs(self, trainloader: Any, number_of_epochs: int) -> None:
        """[summary]

        .. todo:: Advance docstrings (https://xainag.atlassian.net/browse/XP-425)

        Args:
            trainloader (~typing.Any): [description]
            number_of_epochs (int): [description]
        """

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

        for _ in tqdm(
            range(number_of_epochs), desc="Epochs"
        ):  # loop over the dataset multiple times

            for _, data in tqdm(enumerate(trainloader, 0), desc="Batches"):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

    def evaluate_on_test(self, testloader: Any) -> float:
        """[summary]

        .. todo:: Advance docstrings (https://xainag.atlassian.net/browse/XP-425)

        Args:
            testloader (~typing.Any): [description]

        Returns:
            float: [description]
        """

        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)  # pylint: disable=no-member
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * (correct / total)
        return acc
