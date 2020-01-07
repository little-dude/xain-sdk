import numpy as np
from typing import Dict, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm


class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1: nn.Conv2d = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x) :  # type: ignore
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def train_n_epochs(self, trainloader: Any, number_of_epochs: int) -> None:
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

    def flatten_weights(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        flattened: np.ndarray = np.concatenate(
            list(self.state_dict().values()), axis=None
        )
        shapes: np.ndarray = [
            weights.shape for weights in list(self.state_dict().values())
        ]
        indices: np.ndarray = np.cumsum([np.prod(shape) for shape in shapes])
        return flattened, shapes, indices

    def read_from_vector(
        self, indices: np.ndarray, flattened: np.ndarray, shapes: np.ndarray
    ) -> None:
        """

        """
        model_weights = np.split(flattened, indices_or_sections=indices)
        model_weights = [
            np.reshape(weights, newshape=shape)
            for weights, shape in zip(model_weights, shapes)
        ]
        tensors = [torch.from_numpy(a) for a in model_weights]
        new_state_dict: Dict = dict()
        new_state_dict.update(zip(self.state_dict().keys(), tensors))
        self.load_state_dict(new_state_dict)

    def evaluate_on_test(self, testloader: Any) -> float:
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * (correct / total)
        return acc
