# Participant Examples

In this section you can find examples how to run and start participants with Tensorflow Keras or PyTorch and connect them to a running coordinator.

- [Tensorflow Keras example for the SDK Participant implementation](https://xain-sdk.readthedocs.io/en/latest/examples/tensorflow_keras.html)
- [PyTorch example for the SDK Participant implementation](https://xain-sdk.readthedocs.io/en/latest/examples/pytorch.html)


# Start multiple participants in parallel

Once the `xain-sdk` package and an example are installed and a coordinator is up and running you can test the federated learning process quickly by using a preconfigured script. For example, the following command, executed in `/examples`, will start five participants with Keras implementations in parallel:

```shell
sh run_participants.sh tensorflow_keras/example.py 5
```

The first argument specifies the python module containing the `start_participant` function for establishing a connection between participant and coordinator while the second argument defines the number of participants to be started in parallel. Be aware that the current maximum advised number of participants running on the same device (depending on the available hardware resources) is around 20 or 30.
