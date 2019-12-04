# Tensorflow Keras Participant Example

This is an example of a Tensorflow Keras implementation of a `Participant` for federated learning.

We cover the requirements of the [Participant Abstract Base Class](#participant-abstract-base-class), give ideas on how to handle a [Keras Model](#keras-model) and [Keras Data](#keras-data) in the `Participant`, and show how to implement a federated learning [Keras Training Round](#keras-training-round). You can find the complete [Source Code](#source-code) at the end of this document.

The example code makes use of typing to be precise about the expected data types, specifically

```python
from typing import Dict, List, Tuple
import numpy as np
```


## Participant Abstract Base Class

The SDK provides an abstact base class for `Participant`s which can be imported as

```python
from xain_sdk.sdk.participant import Participant as ABCParticipant
```

A custom `Participant` should inherit from the abstract base class, like

```python
class Participant(ABCParticipant):
```

and must implement the `train_round()` method in order to be able to execute a round of federated learning, where each round consists of a certain number of epochs. This method adheres to the function signature

```python
def train_round(
    self, weights: List[np.ndarray], epochs: int, epoch_base: int
) -> Tuple[List[np.ndarray], Dict[str, List[np.ndarray]]]:
```

The expected arguments are:

- `weights (List[np.ndarray])`: A list of Numpy arrays containing the weights of the global model.
- `epochs (int)`: The number of epochs to be trained during the federated learning round.
- `epoch_base (int)`: An epoch base number in case the state of the training optimizer is dependent on the overall epoch (e.g. for learning rate schedules).

The expected return values are:
- `List[np.ndarray]`: The weights of the local model which results from the global model after certain `epochs` of training on local data.
- `Dict[str, List[np.ndarray]]`: The metrics gathered during the training. This might be an empty dictionary if the `Coordinator` is not supposed to collect the metrics.


## Keras Model

A Keras model might either be loaded from a file, generated during the initialization of the `Participant`, or even generated on the fly in a `train_round()`. Here, we present a simple dense neural network for classification generated during the `Participant`'s initialization. We make use of the Keras components

```python
from tensorflow import Tensor
from tensorflow.data import Dataset
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
```

The example model consists of an input layer holding 10 parameters per sample, as

```python
input_layer: Tensor = Input(shape=(10,), dtype="float32")
```

Next, it has a fully connected hidden layer with 6 relu-activated units, as

```python
hidden_layer: Tensor = Dense(
    units=6,
    activation="relu",
    use_bias=True,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
)(inputs=input_layer)
```

Finally, it has a fully connected output layer with 2 softmax-activated units, as

```python
output_layer: Tensor = Dense(
    units=2,
    activation="softmax",
    use_bias=True,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
)(inputs=hidden_layer)
```

The model gets compiled with an Adam optimizer, the categorical crossentropy loss function and the categorical accuracy metric, like

```python
self.model: Model = Model(inputs=[input_layer], outputs=[output_layer])
self.model.compile(
    optimizer="Adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"]
)
```


## Keras Data

The data on which the model will be trained, can either be loaded from a data source (e.g. file, bucket, data base) during the initialization of the `Participant` or on the fly in a `train_round()`. Here, we employ randomly generated placeholder data as an example. This is by no means a meaningful dataset, but it should be sufficient to convey the overall idea. The dataset for training gets shuffled and batched, like

```python
self.trainset: Dataset = Dataset.from_tensor_slices(
    tensors=(np.ones(shape=(80, 10), dtype=np.float32), np.eye(N=80, M=10, dtype=np.float32))
).shuffle(buffer_size=80).batch(batch_size=10)
```

while the datasets for validation and testing only get batched, like

```python
self.valset: Dataset = Dataset.from_tensor_slices(
    tensors=(np.ones(shape=(10, 10), dtype=np.float32), np.eye(N=10, M=10, dtype=np.float32))
).batch(batch_size=10)
self.testset: Dataset = Dataset.from_tensor_slices(
    tensors=(np.ones(shape=(10, 10), dtype=np.float32), np.eye(N=10, M=10, dtype=np.float32))
).batch(batch_size=10)
```


## Keras Training Round

The implementation of the actual `train_round()` method consists of three main steps. First, the provided `weights` of the global model are loaded into the local model, as

```python
self.model.set_weights(weights)
```

Next, a dictionary for the gathered metrics is initialized by

```python
metrics: Dict[str, List[np.ndarray]] = {metric_name: [] for metric_name in self.model.metrics_names}
```

and the local model is trained for certain `epochs` on the local data, whereby the metrics gathered in each epoch are added to the dictionary, as

```python
for _ in range(epochs):
    self.model.fit(x=self.trainset, verbose=2, shuffle=False)
    for metric_name, metric in zip(
        self.model.metrics_names, self.model.evaluate(x=self.valset, verbose=0)
    ):
        metrics[metric_name].append(metric)
```

This explicit training loop is due to the Tensorflow v1 datasets and their handling in `fit()` and `evaluate()`. Hence a Tensorflow v2 approach would reduce to a single call to `fit()` additionally specifying the validation data and number of epochs. The metrics could then be gathered from the history dictionary returned by `fit()`.

Finally, the updated weights of the local model and the gathered metrics are returned, as

```python
return self.model.get_weights(), metrics
```

Additional information might be included in the metrics dictionary, e.g. data distribution of the local dataset.


## Source Code

```python
"""Keras/Tensorflow example for the SDK Participant implementation."""

from typing import Dict, List, Tuple

import numpy as np
from tensorflow import Tensor
from tensorflow.data import Dataset
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense

from xain_sdk.sdk.participant import Participant as ABCParticipant


class Participant(ABCParticipant):
    """An example of a Keras/Tensorflow implementation of a participant for federated learning.

    The attributes for the model and the datasets are only for convenience, they might as well be
    loaded in the `train_round()` method on the fly.

    Attributes:
        model: The model to be trained.
        trainset: A dataset for training.
        valset: A dataset for validation.
        testset: A dataset for testing.
    """

    def __init__(self):
        """Initialize the custom participant.

        The model and the datasets are defined here only for convenience, they might as well be
        loaded in the `train_round()` method on the fly. Due to the nature of this example, the
        model is a simple dense neural network and the datasets are randomly generated.
        """

        super(Participant, self).__init__()

        # define or load a model to be trained
        input_layer: Tensor = Input(shape=(10,), dtype="float32")
        hidden_layer: Tensor = Dense(
            units=6,
            activation="relu",
            use_bias=True,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
        )(inputs=input_layer)
        output_layer: Tensor = Dense(
            units=2,
            activation="softmax",
            use_bias=True,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
        )(inputs=hidden_layer)
        self.model: Model = Model(inputs=[input_layer], outputs=[output_layer])
        self.model.compile(
            optimizer="Adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"]
        )

        # define or load data to be trained on
        self.trainset: Dataset = (
            Dataset.from_tensor_slices(
                tensors=(
                    np.ones(shape=(80, 10), dtype=np.float32),
                    np.eye(N=80, M=10, dtype=np.float32),
                )
            )
            .shuffle(buffer_size=80)
            .batch(batch_size=10)
        )
        self.valset: Dataset = Dataset.from_tensor_slices(
            tensors=(
                np.ones(shape=(10, 10), dtype=np.float32),
                np.eye(N=10, M=10, dtype=np.float32),
            )
        ).batch(batch_size=10)
        self.testset: Dataset = Dataset.from_tensor_slices(
            tensors=(
                np.ones(shape=(10, 10), dtype=np.float32),
                np.eye(N=10, M=10, dtype=np.float32),
            )
        ).batch(batch_size=10)

    def train_round(  # pylint: disable=unused-argument
        self, weights: List[np.ndarray], epochs: int, epoch_base: int
    ) -> Tuple[List[np.ndarray], Dict[str, List[np.ndarray]]]:
        # pylint: disable=line-too-long
        """Train the model in a federated learning round.

        A global model is given in terms of its `weights` and it is trained on local data for a
        number of `epochs`. The weights of the updated local model are returned together with a set
        of metrics.

        Args:
            weights (~typing.List[~numpy.ndarray]): The weights of the global model.
            epochs (int): The number of epochs to be trained.
            epoch_base (int): The epoch base number for the optimizer state (in case of epoch
                dependent optimizer parameters).

        Returns:
            ~typing.Tuple[~typing.List[~numpy.ndarray], ~typing.Dict[str, ~typing.List[~numpy.ndarray]]]:
                The updated model weights and the gathered metrics.
        """
        # pylint: enable=line-too-long

        # load the weights of the global model into the local model
        self.model.set_weights(weights)

        # train the local model for the specified number of epochs and gather the metrics
        metrics: Dict[str, List[np.ndarray]] = {
            metric_name: [] for metric_name in self.model.metrics_names
        }
        for _ in range(epochs):
            self.model.fit(x=self.trainset, verbose=2, shuffle=False)
            for metric_name, metric in zip(
                self.model.metrics_names, self.model.evaluate(x=self.valset, verbose=0)
            ):
                metrics[metric_name].append(metric)

        # return the updated weights of the local model and the gathered metrics
        return self.model.get_weights(), metrics

```
