import tensorflow as tf
import numpy as np
from tensorflow import layers

SIZE_OF_PORTFOLIO_ALLOCATOR_OUTPUT = 0  # given as a tuple
NUM_PL_HIDDEN_LAYERS = 5 # number of power law network hidden layers
NUM_PA_HIDDEN_LAYERS = 7 # number of portfolio allocator hidden layers
SIZE_OF_PL_OUTPUT_LAYER = 30  # However many numbers are required to describe ten different power laws


def make_power_law():
    """
    creates a model which produces a variety of power law distributions which will try to minimize the
    :return:
    """
    model = tf.keras.Sequential()

    # input layer - Portfolio Allocation
    model.add(layers.Dense(300, activation='relu', input_shape=SIZE_OF_PORTFOLIO_ALLOCATOR_OUTPUT))
    model.add(layers.BatchNormalization(0.2))

    # Hidden Layers
    # TODO try different architectures. This is very uniform and perhaps ill suited to the task.
    for _ in range(NUM_PL_HIDDEN_LAYERS):
        model.add(layers.Dense(500, activation='relu'))
        model.add(layers.BatchNormalization())
        # TODO may need to add dropout layers - see the kaggle tutorial

    # output layer
    model.add(layers.Dense(SIZE_OF_PL_OUTPUT_LAYER, activation='relu'))

    return model

def make_portfolio_allocator():
    """
    Creates a portfolio allocator model model which chooses a portfolio given a collection of power_law distributions.
    The chose portfolio tries to minimize losses and maximize gains.
    :return: portfolio allocator model
    """
    model = tf.keras.Sequential()

    model.add(layers.Dense(500, activation='relu', input_shape=SIZE_OF_PL_OUTPUT_LAYER))

    # Hidden layers
    for _ in range(NUM_PA_HIDDEN_LAYERS):
        model.add(layers.Dense(500, activation='relu'))
        model.add(layers.BatchNormalization())

    # output layer
    model.add(layers.Dense(SIZE_OF_PORTFOLIO_ALLOCATOR_OUTPUT, activation='relu'))

    return model

def portfolio_allocator_loss(real_output, fake_output):
    """
    Takes in a vector with the real_output and the fake output and outputs the loss. For the portfolio allocator this
    means calculating the amount of money lost from the portfolio given the distribution
    :param real_output:
    :param fake_output:
    :return: total portfolio loss
    """

    # TODO how to do this? What is the real output and what is the fake output? How to define LOSS?

def distribution_maker_loss(real_output, fake_output):
    """
    takes a portfolio allocation and calculates the loss. How much the portfolio gains and how much we want to it to
    lose.
    :param real_output:
    :param fake_output:
    :return:
    """

    # TODO how to do this? What is the real output and what is the fake output? How to define LOSS?

# Todo Train?