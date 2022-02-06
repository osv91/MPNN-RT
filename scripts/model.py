import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.inchi import *
from rdkit import RDLogger
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.Draw import MolsToGridImage
import logging
from multiprocessing import Pool
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from custom_layers import *

def MPNNModel(
    atom_dim,
    bond_dim,
    batch_size=64,
    message_units=128,
    message_steps=8,
    num_attention_heads=8,
    dense_units=512,
):
    atom_features = layers.Input((atom_dim), dtype="float32", name="atom_features")
    
    bond_features = layers.Input((bond_dim), dtype="float32", name="bond_features")
    
    pair_indices = layers.Input((2), dtype="int32", name="pair_indices")
    
    atom_partition_indices = layers.Input(
        (), dtype="int32", name="atom_partition_indices"
    )

    x = MessagePassing(message_units, message_steps)(
        [atom_features, bond_features, pair_indices]
    )

    x = PartitionPadding(batch_size)([x, atom_partition_indices])

    x = layers.Masking()(x)

    x = TransformerEncoder(num_attention_heads, message_units, dense_units)(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(dense_units, activation="relu")(x)
     
    x = layers.Dense(256, activation="relu")(x)
    
    x = layers.Dense(128, activation="relu")(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(1, activation="linear")(x)

    model = keras.Model(
        inputs=[atom_features, bond_features, pair_indices, atom_partition_indices],
        outputs=[x],
    )
    return model

def MPNNModel_frozen(
    atom_dim,
    bond_dim,
    batch_size=64,
    message_units=128,
    message_steps=8,
    num_attention_heads=8,
    dense_units=512,
):
    atom_features = layers.Input((atom_dim), dtype="float32", name="atom_features")
    
    bond_features = layers.Input((bond_dim), dtype="float32", name="bond_features")
    
    pair_indices = layers.Input((2), dtype="int32", name="pair_indices")
    
    atom_partition_indices = layers.Input(
        (), dtype="int32", name="atom_partition_indices"
    )

    x = MessagePassing(message_units, message_steps)(
        [atom_features, bond_features, pair_indices]
    )

    x = PartitionPadding(batch_size)([x, atom_partition_indices])

    x = layers.Masking()(x)

    x = TransformerEncoder(num_attention_heads, message_units, dense_units)(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(dense_units, activation="relu")(x)
     
    x = layers.Dense(256, activation="relu")(x)
    
    x = layers.Dense(128, activation="relu")(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(1, activation="linear")(x)
    atom_features.trainable=False
    bond_features.trainable=False
    pair_indices.trainable=False
    atom_partition_indices.trainable=False
    MessagePassing.trainable=False
    PartitionPadding.trainable=False
    layers.Masking.trainable = False
    layers.GlobalAveragePooling1D.trainable=False

    model = keras.Model(
        inputs=[atom_features, bond_features, pair_indices, atom_partition_indices],
        outputs=[x],
    )
    return model



