# PyTorch Participant Example

This is an example of a PyTorch implementation of a `Participant` class for Federated Learning.
We follow the example presented in [this tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py) and it is beneficial to read it before starting this tutorial. 

Every example contains two steps:
- Setting up the Coordinator that waits for the Participants 
- Setting up the SDK Participant together with a suitable model that connects to the Coordinator. 

The first part is described in the [XAIN-fl](https://github.com/xainag/xain-fl) repository. Then we can assume that we have our Coordinator waiting for the Participants to join. The next step is to set up the Participant SDK and equip it with a model. We cover the requirements of the [Participant Abstract Base Class](#participant-abstract-base-class), give ideas on how to handle a PyTorch model and show how to implement a Federated Learning example. You can find the complete source code [here](https://github.com/xainag/xain-sdk/blob/master/examples/pytorch/example.py). The example code makes use of typing to be precise about the expected data types.


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
train_round(self, weights: Optional[np.ndarray], epochs: int, epoch_base: int)
-> Tuple[np.ndarray, int, Dict[str, np.ndarray]]
```

The expected arguments are:
- `weights (Optional[np.ndarray])`: Either a Numpy array containing the flattened weights of the global model or None. In the latter case the participant must properly initialize the weights instead of loading them.
- `epochs (int)`: The number of epochs to be trained during the federated learning round. Can be any non-negative number including zero.
- `epoch_base (int)`: An epoch base number in case the state of the training optimizer is dependent on the overall epoch (e.g. for learning rate schedules).

The expected return values are:
- `np.ndarray`: The flattened weights of the local model which results from the global model after certain `epochs` of training on local data.
- `int`: The number of samples in the train dataset used for aggregation strategies.
- `Dict[str, np.ndarray]`: The metrics gathered during the training. This might be an empty dictionary if the `Coordinator` is not supposed to collect the metrics.

The `Participant`'s base class provides utility methods to set the weights of the local model according to the given flat weights vector, by

```python
get_pytorch_weights(model: torch.nn.Module) -> np.ndarray
```

and to get a flattened weights vector from the local model, by

```python
set_pytorch_weights(weights: np.ndarray, shapes: List[Tuple[int, ...]], model: torch.nn.Module) -> None
```

as well as the original shapes of the weights of the local model, by

```python
get_pytorch_shapes(model: torch.nn.Module) -> List[Tuple[int, ...]]
```


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

The last part of our model class is setting the optimizer together with loss function and training loop.

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

that is an instance of the CNN we defined following the PyTorch tutorial. The utility method for setting the model weights require the original shapes of the weights, obtainable as

```python
self.model.forward(torch.zeros((4, 3, 32, 32)))
self.model_shapes: List[Tuple[int, ...]] = self.get_pytorch_shapes(model=self.model)
```

where the dummy forward pass is necessary to populate the state dict but has no effect otherwise.


## PyTorch Training Round

The implementation of the actual `train_round()` method consists of three main steps. First, the provided `weights` of the global model are loaded into the local model, as

```python 
if weights is not None:
    self.set_pytorch_weights(weights=weights, shapes=self.model_shapes, model=self.model)
```

Next, the local model is trained for certain `epochs` on the local data, whereby the metrics are gathered in each epoch, as

```python
number_samples = len(self.trainloader)
self.model.train_n_epochs(self.trainloader, epochs)
```

The metrics are transformed into a dictionary, which maps metric names to the gathered metric values, by

```python
metrics = self.model.evaluate_on_test(self.testloader))
```

Finally, the updated weights of the local model, the number of samples of the train dataset and the gathered metrics are returned, as

```python
weights = self.get_pytorch_weights(model=self.model)
return weights, number_samples, metrics
```

If there are no weights provided, then the participant initializes new weights according to its model definition and returns them without further training, as

```python
else:
    self.init_model()
    number_samples = 0
    metrics = {}
```
