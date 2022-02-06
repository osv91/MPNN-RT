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
from utils import *
from custom_layers import *
from model import *
from plots import *
from evals import *
#remove RDKit warnings
tf.get_logger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

np.random.seed(123)
tf.random.set_seed(123)



#read input data
import configparser
config = configparser.ConfigParser()
config.read('./config.cfg')
def getConfig(section, attribute, default=""):
    try:
        return config[section][attribute]
    except:
        return default
dataset_file = getConfig("Task","dataset")

dataset_name = dataset_file.split("/")[len(dataset_file.split("/"))-1].rstrip('.csv')

rt_threshold = float(getConfig("Task", "rt_min"))
figures_path=getConfig("Task", "figures_path")
model_path = getConfig("Task", "model_path")
results_path = getConfig("Task", "results_path")




pretrained_path = getConfig("Task", "pretrained_path")
transfer_learning = getConfig("Task", "transfer learning")
cv= getConfig("Task", "CV")
if cv=="True":
    n_folds=5
else:
    n_folds=2


#read dataset as dataframe
df = pd.read_csv(dataset_file,sep=';')[['inchi', 'rt']]
print(df.shape)
sns.displot(df.rt, bins=50)
plt.savefig((figures_path+'/'+dataset_name+'_distribution'+'.svg'))
#remove non-retained molecules
df = df[df.rt>=rt_threshold]
print(df.shape)
#generate smiles instead of inchies
df['smiles'] = df.inchi.apply(inchi2smiles)
df=df[df.smiles!='NA']

x=graphs_from_smiles(df.smiles)
y = df.rt.values
#get an independent test set (one for all CV splits)


permuted_indices = np.random.permutation(np.arange(df.shape[0]))
train_index = permuted_indices[: int(df.shape[0] * 0.80)]
test_index = permuted_indices[int(df.shape[0] * 0.80) :]

X = [x[i][train_index]for i in range(len(x))]
x_test = [x[i][test_index] for i in range(len(x))]
Y = y[train_index]
y_test = y[test_index]



from sklearn.model_selection import KFold
kf=KFold(n_splits=n_folds)

callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0.01,
    patience=20,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=True,
)


# snippet of using the ReduceLROnPlateau callback
from keras.callbacks import ReduceLROnPlateau
rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=10)


import tensorflow_probability as tfp
import keras.backend as K
def medAE(y_true, y_pred):
    
    median = tfp.stats.percentile(K.abs(y_pred-y_true), 50.0, interpolation='midpoint')
    return median



i=0

if cv=="True":
    test_dataset = MPNNDataset(x_test, y_test, batch_size=8)
    for train_ind, test_ind in kf.split(X[0]):
        i+=1
        x_train = [X[i][train_ind]for i in range(len(X))]
        x_valid = [X[i][test_ind] for i in range(len(X))]
        y_train = Y[train_ind]
        y_valid = Y[test_ind]
        
        train_dataset = MPNNDataset(x_train, y_train, batch_size=8)
        valid_dataset = MPNNDataset(x_valid, y_valid, batch_size=8)
          
        if transfer_learning=="False":
            mpnn = MPNNModel(atom_dim=x_train[0][0][0].shape[0], bond_dim=x_train[1][0][0].shape[0], batch_size=8)
        else:
            mpnn = MPNNModel_frozen(atom_dim=x_train[0][0][0].shape[0], bond_dim=x_train[1][0][0].shape[0], batch_size=8)
            mpnn.load_weights(pretrained_path)
            
            
            
        for lr in [2e-4]:
            mpnn.compile(
                loss=keras.losses.MeanAbsoluteError(),
                optimizer=keras.optimizers.Adam(learning_rate=lr),
                metrics=[keras.metrics.MeanAbsoluteError(name="MAE"), keras.metrics.RootMeanSquaredError(name='RMSE'), keras.metrics.MeanAbsolutePercentageError(name="MAPE"),  medAE],
               
            )
            
            
            history = mpnn.fit(
            train_dataset,
            validation_data=valid_dataset,
            epochs=300, callbacks = [callback,rlrop],
            verbose=2)
        
        mpnn.save(model_path+ '/mpnn_'+dataset_name+str(i)+'.h5')  

        #get experimental labels
        Y_train = np.concatenate([y for x, y in train_dataset], axis=0)
        Y_valid = np.concatenate([y for x, y in valid_dataset], axis=0)
        Y_test = np.concatenate([y for x, y in test_dataset], axis=0)
          
          
        #get predicted labels
        y_train = tf.squeeze(mpnn.predict(train_dataset), axis=1)    
        y_valid = tf.squeeze(mpnn.predict(valid_dataset), axis=1)        
        y_test = tf.squeeze(mpnn.predict(test_dataset), axis=1)    
        
        evaluate_ds(Y_train, y_train, results_path, dataset_name, "Train",  i)
        evaluate_ds(Y_valid, y_valid, results_path, dataset_name, "Valid",  i)
        evaluate_ds(Y_test, y_test, results_path, dataset_name, "Test",  i)
    
        
        
        
        
      
else:
    test = pd.DataFrame({'inchi':df.inchi.values[test_index], 'rt':df.rt.values[test_index]})
    test.to_csv('../data/test_'+dataset_name+'.csv', sep=';', index=False)
    train_dataset = MPNNDataset(X, Y, batch_size=8)
    valid_dataset = MPNNDataset(x_test, y_test, batch_size=8)
    if transfer_learning=="False":
       mpnn = MPNNModel(atom_dim=X[0][0][0].shape[0], bond_dim=X[1][0][0].shape[0], batch_size=8)
    else:
        mpnn = MPNNModel_frozen(atom_dim=X[0][0][0].shape[0], bond_dim=X[1][0][0].shape[0], batch_size=8)
        mpnn.load_weights(pretrained_path)
            
            
    i=0        
    for lr in [2e-4]:
        i+=1
        mpnn.compile(
        loss=keras.losses.MeanAbsoluteError(),
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            metrics=[keras.metrics.MeanAbsoluteError(name="MAE"), keras.metrics.RootMeanSquaredError(name='RMSE'), keras.metrics.MeanAbsolutePercentageError(name="MAPE"),  medAE],
               
        )
            
            
        history = mpnn.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=300, callbacks = [callback, rlrop],
        verbose=2)
        
        
    
    
    
    
    mpnn.save(model_path+ '/mpnn_'+dataset_name+'_final'+'.h5')  
    
    #get experimental labels
    Y_train = np.concatenate([y for x, y in train_dataset], axis=0)
    Y_valid = np.concatenate([y for x, y in valid_dataset], axis=0)
             
    #get predicted labels
    y_train = tf.squeeze(mpnn.predict(train_dataset), axis=1)    
    y_valid = tf.squeeze(mpnn.predict(valid_dataset), axis=1)        
       
    evaluate_ds(Y_train, y_train, results_path, dataset_name, "Train",  i)
    evaluate_ds(Y_valid, y_valid, results_path, dataset_name, "Valid",  i)
    
    
    plot_learning_curve(history, figures_path,  dataset_name)
    plot_pred_exp(Y_valid, y_valid, figures_path, dataset_name)
    plot_err_dist(Y_valid, y_valid, figures_path, dataset_name)
    MRE_boxplot(Y_valid, y_valid, figures_path, dataset_name)
    
    
    
       
