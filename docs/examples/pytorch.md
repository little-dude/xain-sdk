# PyTorch Participant Example

This is an example of a PyTorch implementation of a `Participant` class for Federated Learning.
We follow the example presented in [this tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py) and it is beneficial to read it before starting this tutorial. 

Every example contains two steps:
- Setting up the Coordinator that waits for the Participants 
- Setting up the SDK Participant together with a suitable model that connects to the Coordinator. 

The first part is described in the [XAIN-fl](https://github.com/xainag/xain-fl) repository. 
Then we can assume that we have our Coordinator waiting for the Participants to join. 
The next step is to set up the Participant SDK and equip it with a model. 
We cover the requirements of the [Participant Abstract Base Class](#participant-abstract-base-class), give ideas on how to handle a PyTorch model and show how to implement a Federated Learning example. You can find the complete source code [here](https://github.com/xainag/xain-sdk/blob/master/examples/pytorch/example.py).

The example code makes use of typing to be precise about the expected data types, specifically

```python
from typing import Dict, List, Tuple
import numpy as np
```


## Participant Abstract Base Class

The SDK provides an abstract base class for `Participant`s which can be imported as

```python
from xain_sdk.participant import Participant as ABCParticipant
```

A custom `Participant` should inherit from the abstract base class, like

```python
class Participant(ABCParticipant):
```

and must implement the `train_round()` method in order to be able to execute a round of Federated Learning, where each round consists of a certain number of epochs. This method adheres to the function signature

```python
def train_round(
    self, weights: List[np.ndarray], epochs: int, epoch_base: int
) -> Tuple[List[np.ndarray], int, Dict[str, np.ndarray]]:
```

The expected arguments are:
- `weights (List[np.ndarray])`: Either a list of Numpy arrays containing the weights of the global model or an empty list. In the latter case the participant must properly initialize the weights instead of loading them.
- `epochs (int)`: The number of epochs to be trained during the federated learning round. Can be any non-negative number including zero.
- `epoch_base (int)`: An epoch base number in case the state of the training optimizer is dependent on the overall epoch (e.g. for learning rate schedules).

The expected return values are:
- `List[np.ndarray]`: The weights of the local model which results from the global model after certain `epochs` of training on local data.
- `int`: The number of samples in the train dataset used for aggregation strategies.
- `Dict[str, np.ndarray]`: The metrics gathered during the training. This might be an empty dictionary if the `Coordinator` is not supposed to collect the metrics.


## PyTorch Model 

Following the tutorial in the PyTorch documentation we prepare the `class Net(nn.Module)` that contains the model architecture, definition of the forward pass together with optimization round. What is more we include two methods that allows us to enforce model weights to the model and export them after training rounds.


### CNN Setup

We use the architecture used in the PyTorch documentation for the CIFAR10 benchmark. We use the following PyTorch packages

```python
from torch import utils
from torchvision import datasets, transforms
```

to work with the PyTorch CNN packages. We start with preparation of our CNN architecture

```python
def __init__(self) -> None:
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(3, 6, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.fc1 = nn.Linear(16 * 5 * 5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

def forward(self, x: Tensor) -> Tensor:
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 16 * 5 * 5)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x
```

In order to import and export model weights we need to set up two methods that allow import/export to a numpy array.

```python 
def read_from_vector(
    self, indices: np.ndarray, flattened: np.ndarray, shapes: np.ndarray
) -> None:
```

and

```python 
def flatten_weights(self) -> [np.ndarray, np.ndarray, np.ndarray]: 
```

Note, that the functions for flattening and expanding weights deal with a single numpy array instead of a list of numpy arrays, that means the flattened weights have to be un-/packed from/to a list in the `train_round()` method. The last part of our model class is setting the optimizer together with loss function and training loop.

```python
def train_n_epochs(self, trainloader, number_of_epochs: int) -> None:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

    for epoch in tqdm(range(number_of_epochs), desc="Epochs"):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in tqdm(enumerate(trainloader, 0), desc="Batches"):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = self(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
```

As we can expect the last function is exactly what we need in our Participant class as the `train_round()` method.


## PyTorch data loader and Participant initialization 

In the tutorial we use the standard data transformation and loading as it is described in the PyTorch documentation. The only difference is that we initiate each Participant with a randomised training set.

```python 
self.trainset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
self.trainloader = utils.data.DataLoader(
    self.trainset, batch_size=4, shuffle=True, num_workers=2
)
```

For validation purposes we also create a test set

```python
self.testset = datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)
self.testloader = utils.data.DataLoader(
    self.testset, batch_size=4, shuffle=False, num_workers=2
)
```

Besides the test and train set, the Participant also contains a model

```python
self.model = Net()
```

that is an instance of the CNN we defined following the PyTorch tutorial. From the model we also obtain its initial weights and shape of the layers.

```python
self.flattened, self.shapes, self.indices = self.model.flatten_weights()
self.number_samples: int = len(self.trainloader)
```

These parameters become handy when we need to import parameters from a single Numpy array. Such an initiated Participant is required to have the `train_round` method to communicate with the Coordinator.


## PyTorch Training Round

The implementation of the actual `train_round()` method consists of three main steps. First, the provided `weights` of the global model are loaded into the local model or they have to be initialized if no weights are present. Then some number of epochs is trained and as a last step we export the weights to a numpy array.

```python
def train_round(
    self, weights: List[np.ndarray], epochs: int, epoch_base: int
) -> Tuple[List[np.ndarray], int, Dict[str, np.ndarray]]:
```

As mentioned before we want to perform three crucial steps in this method

```python 
if weights:
    self.model.read_from_vector(self.indices, weights[0], self.shapes)
    self.model.train_n_epochs(self.trainloader, epochs)
    self.flattened, self.shapes, self.indices = self.model.flatten_weights()
```

which we defined together with the CNN class. The last step is to calculate matrices that we are interested in our model.

```python
number_samples = len(self.trainloader)
metrics = self.model.evaluate_on_test(self.testloader))
```

If there are no weights provided, then the participant initializes new weights according to its model definition and returns them without further training, as

```python
else:
    self.init_model()
    number_samples = 0
    metrics = {}
```
