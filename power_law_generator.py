import tensorflow as tf
import numpy as np
from tensorflow import layers
import portfolio_allocator

SIZE_POWER_LAW_GENERATOR = (10)
NUM_PL_HIDDEN_LAYERS = 5 # number of power law network hidden layers
SIZE_OF_PL_OUTPUT_LAYER = 30  # However many numbers are required to describe ten different power laws



class DistributionGenerator:
    def __init__(self, allocation=[]):
        """
        Creates a distribution generator model.
        :param allocation: a numpy array describing the allocation of the portfolio.
        """
        self.portfolio_allocation = allocation

    def make_power_law(self):
        """
        creates a model which produces a variety of power law distributions which will try to minimize the
        :return:
        """
        model = tf.keras.Sequential()

        # input layer - Portfolio Allocation
        model.add(layers.Dense(300, activation='relu',
                               input_shape=portfolio_allocator.SIZE_OF_PORTFOLIO_ALLOCATOR_OUTPUT))
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



    def portfolio_allocator_loss(self, real_output, fake_output):
        """
        Takes in a vector with the real_output and the fake output and outputs the loss. For the portfolio allocator this
        means calculating the amount of money lost from the portfolio given the distribution. Using the geometric
        distribution will make this more realistic.
        :param real_output:
        :param fake_output:
        :return: total portfolio loss
        """

        # TODO how to do this? What is the real output and what is the fake output? How to define LOSS?
        # The best way to structure this I suspect is to seperate the seperate modules into different classes. There
        # needs to be a global variable, which can be changed. So when we are training the portfolio allocator,
        # we have to keep the vector describing the output, this is what we need in order to calculate the loss.



        # we want to bring this value to zero, so any amount of success is anathema to the dist generator and we want
        # to make it smaller
        loss = fake_output*self.portfolio_allocation
        geometric_average = np.exp(np.log(loss).mean())

        return geometric_average

    def calculate_losses(self, distribution, distribution_class="binom"):
        """
        calculates the probability of losses from the distribution. i.e. we are given ten distributions, what is the
        probability of loss from each one based on the type of distribution
        :param distribution: the vector describing the distribution
        :param distribution_class: the type of distribution
        :return: the vector describing the losses
        """

        #TODO python lacks a switch statement. Therefore, we have to create a dictionary. Each string value should
        # match to its own function? But that will only be relevant later




# Todo Train?