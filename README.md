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


## Getting started

### Run the XAIN Coordinator

There are two options to run the XAIN Coordinator to perform Federated Learning on locally trained models: 

- Go to the main page of the project and request a demo for the [XAIN Platform](https://www.xain.io/federated-learning-platform).
- For the self-hosted solution, see [XAIN FL Project](https://github.com/xainag/xain-fl) for more details.


### Integrate the XAIN SDK into your project

#### 1. Install the XAIN SDK

To install the XAIN SDK package on your machine, simply run in your terminal:

```bash
pip install xain-sdk
```


#### 2. Register your application and the device to participate in the aggregation

Now you can register your Participants to participate in the Federated Learning rounds. To do so, 
just send the registration request to the XAIN Coordinator:


###### participant.py

```python
from typing import Dict, List, Tuple
from numpy import ndarray
from xain_sdk.participant import Participant

class MyParticipant(Participant):
    def __init__(self):

        super(MyParticipant, self).__init__()

        # define or load a model to be trained
        ...

        # define or load data to be trained on
        ...

    def train_round(
        self, weights: List[ndarray], epochs: int, epoch_base: int
    ) -> Tuple[List[ndarray], int, Dict[str, ndarray]]:
        
        # define the number of samples in the training dataset
        number_train_samples: int = 80
        
        # load weights into the model
        ...

        # train the model for the specified number of epochs
        ...
        
        # gather the metrics of the trained epochs
        ...

        # return the updated weights
        return weights, number_train_samples, metrics
```


###### start.py

```python
from xain_sdk.participant_state_machine import start_participant

# Import MyParticipant from your participant.py file 
from participant import MyParticipant

# Create a new participant
p = MyParticipant()

# Register your new participant to interact with XAIN Coordinator (hosted at XAIN Platform or self-hosted solution). The function start_participant requires two arguments:
#   - your new participant to register to interact with Coordinator,
#   - the URL of the Coordinator to connect to. 
start_participant(p, "your_host:your_port")
```

Now you have registered a participant. Simply repeat this step for all the participants you wish to register.

The XAIN Coordinator will take care of the rest: 
- The aggregation of your locally pretrained models.
- Triggering new training rounds for the selected participants and aggregating these models.


#### Model metrics

A monitoring feature, which will be available as a [XAIN Platform solution](https://www.xain.io/federated-learning-platform). If you would like to compare the performance of aggregated models, please send the specific metrics of your use case that you wish to monitor to the XAIN Coordinator. This will then be reflected in the web interface under the `Project Management` tab. In order to send your metrics to the XAIN Coordinator, you will need to update the `train_round()` method accordingly.

The returned `metrics` dictionary expects `str`s as chosen names of the metrics, which are mapped to `ndarray`s as values of the metrics. The first dimension of the `ndarray` should refer to the trained `epochs` of the current round and the following dimensions to the metric values per epoch. For example, a scalar-sized metric like the loss will result in a `ndarray` of shape `(epochs, 1)` and a vector-sized metric like accuracy per category will result in a `ndarray` of shape `(epochs, number_of_categories)`. But in general, any metric of any positive dimension will be accepted. The `metrics` object then might for example look like

```python
metrics = {
    "loss": loss_array,
    "accuracy": accuracy_array,
    "accuracy_per_category": accuracy_per_category_array,
}
```


## Examples

Please see the following examples showing how to implement your Participant with the SDK:
- [Keras/Tensorflow example for the SDK Participant implementation](https://xain-sdk.readthedocs.io/en/latest/examples/tensorflow_keras.html)
- [PyTorch example for the SDK Participant implementation](https://xain-sdk.readthedocs.io/en/latest/examples/pytorch.html)


## Getting help

If you get stuck, don't worry, we are here to help. You can find more information here:

- [More information about the project](https://docs.xain.io)
- [SDK Documentation](https://xain-sdk.readthedocs.io/en/latest/)
- [GitHub issues](https://github.com/xainag/xain-sdk/issues)
- [More information about XAIN](https://xain.io)

In case you don't find answers to your problems or questions, simply ask your question to the community here:

- [Gitter XAIN PROJECT](https://gitter.im/xainag)
