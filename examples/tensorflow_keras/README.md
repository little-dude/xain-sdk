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

Follow the documentation on [xain-fl](https://github.com/xainag/xain-fl/tree/master#running-the-coordinator-locally) 
to start a new coordinator. Set the CLI argument `-f` to the path of the `weights.npy` file.

**Start a participant**

```shell
python example.py
```
