# Tensorflow Keras Participant Example

This is an example of a Tensorflow Keras implementation of a `Participant` for federated learning.

We cover the requirements of the [Participant Abstract Base Class](#participant-abstract-base-class), give ideas on how to handle a [TF Keras Model](#tf-keras-model) and [TF Keras Data](#tf-keras-data) in the `Participant`, and show how to implement a federated learning [TF Keras Training Round](#tf-keras-training-round). You can find the complete source code [here](https://github.com/xainag/xain-sdk/blob/master/examples/tensorflow_keras/example.py).

The example code makes use of typing to be precise about the expected data types, specifically

```python
from typing import Dict, List, Tuple
import numpy as np
```


## Participant Abstract Base Class

The SDK provides an abstact base class for `Participant`s which can be imported as

```python
from xain_sdk.participant import Participant as ABCParticipant
```

A custom `Participant` should inherit from the abstract base class, like

```python
class Participant(ABCParticipant):
```

and must implement the `train_round()` method in order to be able to execute a round of federated learning, where each round consists of a certain number of epochs. This method adheres to the function signature

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


## TF Keras Model

A TF Keras model might either be loaded from a file, generated during the initialization of the `Participant`, or even generated on the fly in a `train_round()`. Here, we present a simple dense neural network for classification generated during the `Participant`'s initialization. We make use of the TF Keras components

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


## TF Keras Data

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


## TF Keras Training Round

The implementation of the actual `train_round()` method consists of three main steps. First, the provided `weights` of the global model are loaded into the local model, as

```python
if weights:
    self.model.set_weights(weights)
```

Next, the local model is trained for certain `epochs` on the local data, whereby the metrics are gathered in each epoch, as

```python
number_samples = 80
metrics_per_epoch: List[List[np.ndarray]] = []
for _ in range(epochs):
    self.model.fit(x=self.trainset, verbose=2, shuffle=False)
    metrics_per_epoch.append(self.model.evaluate(x=self.valset, verbose=0))
```

The metrics are transformed into a dictionary, which maps metric names to the gathered metric values, by

```python
metrics = {
    name: np.stack(np.atleast_1d(*metric))
    for name, metric in zip(self.model.metrics_names, zip(*metrics_per_epoch))
}
```

This explicit training loop is due to the Tensorflow v1 datasets and their handling in `fit()` and `evaluate()`. Hence a Tensorflow v2 approach would reduce to a single call to `fit()` additionally specifying the validation data and number of epochs. The metrics could then for example be gathered from the history dictionary returned by `fit()`.

Finally, the updated weights of the local model, the number of samples of the train dataset and the gathered metrics are returned, as

```python
return self.model.get_weights(), number_samples, metrics
```

If there are no weights provided, then the participant initializes new weights according to its model definition and returns them without further training, as

```python
else:
    self.init_model()
    number_samples = 0
    metrics = {}
```
