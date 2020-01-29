# Tensorflow Keras Participant Example

You can find the explanation of the example [here](https://xain-sdk.readthedocs.io/en/latest/examples/tensorflow_keras.html).


## Install

```shell
# If you don't have the xain-sdk package installed on your system yet, install it with pip first.
pip install -e .
cd examples/tensorflow_keras

# Install the requirements to run the tensorflow keras example.
pip install -r requirements.txt
```


## Run

**Start a coordinator**

Follow the documentation on [xain-fl](https://github.com/xainag/xain-fl/tree/master#running-the-coordinator-locally) to start a new coordinator.

**Start a storage service**

The participants upload their results to an S3 bucket at the end of
each round, so they need to be configured to connect to the relevant
S3 service. Follow the documentation in the [`xain-fl`
project](https://github.com/xainag/xain-fl/tree/master#running-the-coordinator-locally)
to start a local `minio` service.

**Start a participant**

```shell
python example.py
```
