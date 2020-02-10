[![Workflow Lint and test (master)](https://github.com/xainag/xain-sdk/workflows/Lint%20and%20test%20%28master%29/badge.svg)](https://github.com/xainag/xain-sdk)
[![PyPI](https://img.shields.io/pypi/v/xain-sdk)](https://pypi.org/project/xain-sdk/)
[![GitHub license](https://img.shields.io/github/license/xainag/xain-sdk)](https://github.com/xainag/xain-sdk/blob/master/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/xain-sdk/badge/?version=latest)](https://xain-sdk.readthedocs.io/en/latest/)
[![Gitter chat](https://badges.gitter.im/xainag.png)](https://gitter.im/xainag)


# XAIN SDK

## Overview

The XAIN project is building a privacy layer for machine learning so that AI projects can meet compliance such as GDPR and CCPA. The approach relies on Federated Learning as enabling technology that allows production AI applications to be fully privacy compliant.

Federated Learning also enables different use-cases that are not strictly privacy related such as connecting data lakes, reaching higher model performance in unbalanced datasets and utilising AI models on the edge.

**The main components:**
- *Coordinator:* The entity that manages all aspects of the execution of rounds for Federated Learning. This includes the registration of clients, the selection of participants for a given round, the determination of
whether sufficiently many participants have sent updated local models, the computation of an aggregated 
global model, and the sending of the latter model to storage or other entities.
- *Participant:* An entity that is the originator of a local dataset that can be selected for local training in the Federated Learning. 
- *Selected Participant:* A Participant that has been selected by the Coordinator to participate in the next or current round.
- *SDK:* The library which allows Participants to interact with the Coordinator and the XAIN Platform.

The source code in this project implements the XAIN SDK to provide your local application a way to communicate with the XAIN Coordinator.


## Getting Started

### Run The XAIN Coordinator

There are two options to run the XAIN Coordinator to perform Federated Learning on locally trained models: 

- Go to the main page of the project and request a demo for the [XAIN Platform](https://www.xain.io/federated-learning-platform).
- For the self-hosted solution, see [XAIN FL Project](https://github.com/xainag/xain-fl) for more details.


### Integrate The XAIN SDK Into Your Project

#### 1. Install The XAIN SDK

To install the XAIN SDK package on your machine, simply run in your terminal:

```bash
pip install xain-sdk
```


#### 2. Register Your Application And The Device To Participate In The Aggregation

Now you can register your Participants to participate in the Federated Learning rounds. To do so, 
just send the registration request to the XAIN Coordinator:


###### participant.py

```python
from typing import Optional, Tuple
from numpy import ndarray
from xain_sdk.participant import Participant

class MyParticipant(Participant):
    def __init__(self):

        super(MyParticipant, self).__init__()

        # define or load a model to be trained
        self.model = ...
        self.model_shapes = self.get_<MLframework>_shapes(self.model)

        # define or load data to be trained on
        self.trainset = ...

    def train_round(
        self, weights: Optional[ndarray], epochs: int, epoch_base: int
    ) -> Tuple[ndarray, int]:

        if weights is not None:
            # load weights into the model
            self.set_<MLframework>_weights(weights, self.model_shapes, self.model)

            # train the model for the specified number of epochs
            number_samples = ...
            for epoch in range(epochs):
                ... # training
                self.update_metrics(epoch, epoch_base, MetricName=metric_value, ...)

        else:
            # initialize new weights for the model
            ...

        # return the updated weights and the number of training samples
        weights = self.get_<MLframework>_weights(self.model)
        return weights, number_samples
```


###### start.py

```python
from xain_sdk.participant_state_machine import start_participant

# Import MyParticipant from your participant.py file 
from participant import MyParticipant

# Create a new participant
p = MyParticipant()

# Register your new participant to interact with XAIN Coordinator 
# (hosted at XAIN Platform or self-hosted solution). 
# The function start_participant requires two arguments:
#   - your new participant to register to interact with Coordinator,
#   - a valid configuration (an example configuration can be found in config/example-config.toml).
config = Config.load("my_config.toml")
start_participant(p, config)
```

Now you have registered a participant. Simply repeat this step for all the participants you wish to register.

The XAIN Coordinator will take care of the rest: 
- The aggregation of your locally pretrained models.
- Triggering new training rounds for the selected participants and aggregating these models.


#### Utility

The `Participant` base class provide some utility methods to help with the implementation of the `train_round()` method, namely:
- `set_tensorflow_weights()`: Set the weights of a Tensorflow model from a flat weight vector.
- `get_tensorflow_weights()`: Get and flatten the weights of a Tensorflow model.
- `get_tensorflow_shapes()`: Get the shapes of the weights of a Tensorflow model.
- `set_pytorch_weights()`: Set the weights of a Pytorch model from a flat weight vector.
- `get_pytorch_weights()`: Get and flatten the weights of a Pytorch model.
- `get_pytorch_shapes()`: Get the shapes of the weights of a Pytorch model.
- `update_metrics()`: Gather metrics to be send to the coordinator.


#### Model Metrics

A monitoring feature, which will be available as a [XAIN Platform solution](https://www.xain.io/federated-learning-platform). If you would like to compare the performance of aggregated models, please send the specific metrics of your use case that you wish to monitor to the XAIN Coordinator. This will then be reflected in the web interface under the `Project Management` tab.

In order to send your metrics to the XAIN Coordinator, you can simply add the following line to the training loop in the `train_round()` method

```python
self.update_metrics(epoch, epoch_base, MetricName=metric_value, ...)
```

where `epoch` is the current local epoch number, `epoch_base` is the global epoch number and `MetricName=metric_value` can be any number of key-value pairs with a name of a metric as key and a numerical scalar/list/array as value.


## Examples

Please see the following examples showing how to implement your Participant with the SDK:
- [Keras/Tensorflow example for the SDK Participant implementation](https://xain-sdk.readthedocs.io/en/latest/examples/tensorflow_keras.html)
- [PyTorch example for the SDK Participant implementation](https://xain-sdk.readthedocs.io/en/latest/examples/pytorch.html)


### Testing

You can connect multiple participants at once running in parallel to a coordinator with the following script:

[Bash script for starting multiple participants](https://github.com/xainag/xain-sdk/tree/master/examples#start-multiple-participants-in-parallel)


## Getting help

If you get stuck, don't worry, we are here to help. You can find more information here:

- [More information about the project](https://docs.xain.io)
- [SDK Documentation](https://xain-sdk.readthedocs.io/en/latest/)
- [GitHub issues](https://github.com/xainag/xain-sdk/issues)
- [More information about XAIN](https://xain.io)

In case you don't find answers to your problems or questions, simply ask your question to the community here:

- [Gitter XAIN PROJECT](https://gitter.im/xainag)
