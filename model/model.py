#!/bin/env python3

import argparse
import numpy as np
import pickle
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model

from create_data import create_data

TSLENGTH = 10
INPUT_LENGTH = 7
OUTPUT_LENGTH = 3
TRAINING_FRACTION = 0.75
TEST_FRACTION = 0.25
NUMBER_OF_EPOCHS = 1
BATCH_SIZE = 1
ERR_THRESHOLD = 0.7


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        help="mode to run the model. options: [train, test]")
    args = parser.parse_args()
    if args.mode == 'train':
        print("Creating data....")
        create_data()
        print("Fitting Model...")
        model = train_model(nb_epoch=NUMBER_OF_EPOCHS, nb_batch=BATCH_SIZE)
    elif args.mode == 'test':
        model = load_model('lstm.h5')
        predictions = evaluate_predictions(model)
        plot_predictions(predictions)


def split_data(type, inputData, labelData):
    numOfRows = inputData.shape[0]

    if type == 'training':
        trainX = inputData[:int(numOfRows * TRAINING_FRACTION)]
        trainY = labelData[:int(numOfRows * TRAINING_FRACTION)]
        return trainX, trainY

    if type == 'test':
        testX = inputData[int(numOfRows * TRAINING_FRACTION):]
        testY = labelData[int(numOfRows * TRAINING_FRACTION):]
        return testX, testY


def train_model(nb_epoch, nb_batch):
    dfX = np.genfromtxt('../raw_input.csv', delimiter=',')
    dfY = np.genfromtxt('../raw_label.csv', delimiter=',')

    # Sanity check
    print("Input Shape: " + str(dfX.shape))
    print("Label shape: " + str(dfY.shape))

    trainX, trainY = split_data('training', dfX, dfY)

    # reshape data from [samples, features] to [samples, timesteps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

    model = Sequential()
    model.add(
        LSTM(
            1,
            batch_input_shape=(
                nb_batch,
                trainX.shape[1],
                trainX.shape[2]),
            stateful=True))
    model.add(Dense(OUTPUT_LENGTH))
    model.compile(loss='mean_squared_error', optimizer='adam')

    for i in range(nb_epoch):
        model.fit(
            trainX,
            trainY,
            epochs=1,
            batch_size=1,
            verbose=1,
            shuffle=False)
        model.reset_states()

    model.save('lstm.h5')
    return model


def predictUsingModel(model, sample):
    """Reshape input pattern to [samples, timesteps, features]
    """
    sample = sample.reshape(1, 1, len(sample))
    # make forecast
    prediction = model.predict(sample, batch_size=BATCH_SIZE)
    # convert to array
    return [x for x in prediction[0, :]]


def evaluate_predictions(model):
    dfX = np.genfromtxt('../raw_input.csv', delimiter=',')
    dfY = np.genfromtxt('../raw_label.csv', delimiter=',')

    file = open("../data.meta", 'rb')
    [min, max] = pickle.load(file)
    file.close()

    testX, testY = split_data('test', dfX, dfY)
    predictions = list()
    # make prediction
    for sample in testX:
        prediction = predictUsingModel(model, sample)
        predictions.append(prediction)

    predictions = inverse_transform(predictions, min, max)

    # compute avg error:
    rmse = 0
    for i in range(testY.shape[0]):
        rmse += sqrt(mean_squared_error(testY[i, :], predictions[i]))

    print('RMSE:' + str(rmse / testY.shape[0]))

    return predictions


def inverse_transform(predictions, min, max):
    return ((np.array(predictions) + 1) * (max - min) / 2) + min


def plot_predictions(predictions):
    """Plot the entire dataset in blue
    """
    dfX = np.genfromtxt('../raw_input.csv', delimiter=',')
    dfY = np.genfromtxt('../raw_label.csv', delimiter=',')

    testX, testY = split_data('test', dfX, dfY)

    for i in range(len(predictions)):
        err = sqrt(mean_squared_error(testY[i, :], predictions[i]))
        if err < ERR_THRESHOLD:
            plt.plot(np.append(testX[i, :], predictions[i]), color='red')
            plt.plot(np.append(testX[i, :], testY[i, :]))

        plt.show()


if __name__ == "__main__":
    main()
