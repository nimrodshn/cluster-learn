#!/bin/env python3

import os
import pandas as pd
import numpy as np

TSLENGTH = 10
INPUT_LENGTH = 7
DATA_PATH = '../data/'
OUTPUT_PATH = '../'
COL_NAME = '\tCPU usage [%]'
MAX_DATA_FILES = 100


def create_data(
        data_path=DATA_PATH,
        output_path=OUTPUT_PATH,
        col_name=COL_NAME,
        tslength=TSLENGTH,
        input_length=INPUT_LENGTH,
        max_data_files=MAX_DATA_FILES):
    """Create a time series dataset from the fastStorage dataset. (See README)
    """
    files = os.listdir(data_path)
    dataX = []
    dataY = []

    for filename in files[:max_data_files]:
        path = data_path + filename

        print("opening file:  " + path)
        df = pd.read_csv(path, sep=';', header=0)
        df = df[col_name]
        numOfRows = df.shape[0]
        numOfOutputRows = numOfRows - TSLENGTH

        for idx in range(numOfOutputRows):
            print("  %s [%05d:%05d]" % (filename, idx, numOfOutputRows))

            dataX.append(df[idx:idx + INPUT_LENGTH])
            dataY.append(df[idx + INPUT_LENGTH:idx + TSLENGTH])

    dfX = np.asarray(dataX)
    dfY = np.asarray(dataY)

    # Sanity check
    print("Input shape: %s" % str(dfX.shape))
    print("Label shape: %s" % str(dfY.shape))

    np.savetxt(output_path + 'raw_input.csv', dfX, delimiter=",")
    np.savetxt(output_path + 'raw_label.csv', dfY, delimiter=",")


if __name__ == "__main__":
    create_data()
