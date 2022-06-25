import numpy as np
import tensorflow as tf
import power_law_generator
from keras import layers

SIZE_OF_PORTFOLIO_ALLOCATOR_OUTPUT = (10)  # given as a tuple
NUM_PA_HIDDEN_LAYERS = 7 # number of portfolio allocator hidden layers


class PortfolioAllocator:

    def __init__(self, dist=[]):
        """
        Creates a Portfolio Allocator class
        :param dist: a numpy array describing the parameters of the distributions
        """
        self.distribution = dist


    def make_portfolio_allocator_model(self):
        """
        Creates a portfolio allocator model model which chooses a portfolio given a collection of power_law distributions.
        The chose portfolio tries to minimize losses and maximize gains.
        :return: portfolio allocator model
        """
        model = tf.keras.Sequential()

        model.add(layers.Dense(500, activation='relu', input_shape=power_law_generator.SIZE_OF_PL_OUTPUT_LAYER))

        # Hidden layers
        for _ in range(NUM_PA_HIDDEN_LAYERS):
            model.add(layers.Dense(500, activation='relu'))
            model.add(layers.BatchNormalization())

        # output layer
        model.add(layers.Dense(SIZE_OF_PORTFOLIO_ALLOCATOR_OUTPUT, activation='relu'))

        return model

    def distribution_maker_loss(self, real_output, fake_output):
        """
        takes a portfolio allocation and calculates the loss. How much the portfolio gains and how much we want to it to
        lose.
        :param real_output:
        :param fake_output:
        :return:
        """

        non_allocated, fake_output = fake_output[0], fake_output[1:]


        # TODO how to do this? What is the real output and what is the fake output? How to define LOSS?

    #Todo training?