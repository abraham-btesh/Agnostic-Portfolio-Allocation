import power_law_generator
import portfolio_allocator
import numpy as np
import tensorflow
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam

INIT_LR = 2e-4
NUM_EPOCHS = 90


# Make the portfolio generator
portfolio = portfolio_allocator.PortfolioAllocator().make_portfolio_allocator_model()

# Make The distribution generator
pl_gen = power_law_generator.DistributionGenerator().make_power_law()

# Compile the portfolio generator
pl_gen.trainable = False
discOpt = Adam(lr=INIT_LR, beta_1=0.5, decay=INIT_LR / NUM_EPOCHS)
portfolio.compile(loss=portfolio_allocator.PortfolioAllocator.distribution_maker_loss, optimizer=discOpt)



# Compile the Distribution Generator
portfolio.trainable = False # hold the portfolio constant
ganOpt = Adam(lr=INIT_LR, beta_1=0.5, decay=INIT_LR / NUM_EPOCHS)
gan.compile(loss=power_law_generator.DistributionGenerator.portfolio_allocator_loss, optimizer=discOpt)







