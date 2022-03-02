import tensorflow as tf

from __future__ import annotations

import numpy as np


from keras.layers import (
    Dense,
    Dropout,
    Reshape,
    Flatten,
    InputLayer,
    Conv1D,
    GlobalMaxPooling1D,
)
from keras.losses import KLDivergence
from keras.models import Sequential, load_model
from keras.models import Model
from keras import metrics
import tensorflow.keras.optimizers as optimizers
import tensorflow as tf
from tensorflow.python.keras.backend import (
    binary_crossentropy,
    conv1d,
    categorical_crossentropy,
)
from tensorflow.python.keras.layers.core import Activation
from tensorflow.python.keras.layers.pooling import MaxPool1D
from tensorflow.python.keras.losses import BinaryCrossentropy

from tcn import TCN