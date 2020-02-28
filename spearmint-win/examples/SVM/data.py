import os
import sklearn
import numpy as np
import pandas as pd
from sklearn import model_selection
import pickle as pkl

URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/molecular-biology/promoter-gene-sequences/promoters.data'
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
OUT_DIR = os.path.join(CURRENT_DIR, 'data')


def get_data(test_size=0.5):
    output_file = os.path.join(OUT_DIR, "test_size_{}".format(test_size))
    if not os.path.exists(output_file):
        generate_and_write_data(test_size)
    with open(output_file, 'rb') as f:
        data = pkl.load(f)
    return data

def generate_and_write_data(test_size=0.5):
    names = ['Class', 'id', 'Sequence']
    data = pd.read_csv(URL, names=names)
    # Turn into a pandas DF where each row is a DNA sequence with 57 nucleotides
    classes = data.loc[:, 'Class']
    sequences = list(data.loc[:, 'Sequence'])
    dataset = {}

    # loop through sequences and split into individual nucleotides
    for i, seq in enumerate(sequences):
        # split into nucleotides, remove tab characters
        nucleotides = [x for x in seq if x != '\t']
        # append class assignment
        nucleotides.append(classes[i])
        # add to dataset
        dataset[i] = nucleotides
    df = pd.DataFrame(dataset)
    df = df.transpose()
    df.rename(columns={57: 'Class'}, inplace=True)
    # Convert to one hot encoding for each nucleotide. Drop the extra class, we only need 1
    numerical_df = pd.get_dummies(df)
    df = numerical_df.drop(columns=['Class_-'])
    df.rename(columns={'Class_+': 'Class'}, inplace=True)

    # Create X and Y datasets for training
    X = np.array(df.drop(['Class'], 1))
    y = np.array(df['Class'])

    # Make each example into a tuple of a single feature vector and an empty edge list
    X_ = [(np.atleast_2d(x), np.empty((0, 2), dtype=np.int)) for x in X]
    Y = y.reshape(-1, 1)
    # split data into training and testing datasets
    data_split = model_selection.train_test_split(X_, Y, test_size=test_size)

    output_file = os.path.join(OUT_DIR, "test_size_{}".format(test_size))
    with open(output_file, 'wb') as f:
        pkl.dump(data_split, f)
