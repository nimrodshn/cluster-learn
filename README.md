# cluster-learn:

cluster-learn is a deep learning model that predicts machine metrics.

The model is trained on a public data set (fastStorage[1]) containing metrics
recorded from 1,250 VMs used by Netherlands major banks, credit card, insurance
and other companies.

For the sake of this POC I have chosen to model CPU Usage [%].
In total there are ~ 1.1 million series in the data set -
Split into 75% (~800,000) training set, 25% test set (~200,000).

[1]http://gwa.ewi.tudelft.nl/datasets/gwa-t-12-bitbrains

## Running the model
The model has two modes - "train" and "test":
```
python3 model.py --mode MODE
```

## Some cool screenshots:
![Figure0](https://github.com/nimrodshn/cluster-learn/blob/master/figures/result11.png)
![Figure1](https://github.com/nimrodshn/cluster-learn/blob/master/figures/result9.png)
![Figure2](https://github.com/nimrodshn/cluster-learn/blob/master/figures/result2.png)
