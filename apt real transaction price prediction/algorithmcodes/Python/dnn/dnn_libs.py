from keras.layers.core import Dense, Activation, Dropout
from keras.models import Sequential

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
import time
