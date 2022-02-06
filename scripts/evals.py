from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score, mean_squared_error
import numpy as np

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def evaluate_ds(Y_true, y_pred, results_path, dataset_name, eval_type,  i=0):
    with open (results_path+'/'+dataset_name+'_'+str(i)+eval_type+'.txt', "w") as f:
        f.write('MAE:'+str(mean_absolute_error(Y_true, y_pred))+'\n')
        f.write('MedAE:'+str(median_absolute_error(Y_true, y_pred))+'\n')
        f.write('RMSE:'+str(mean_squared_error(Y_true, y_pred, squared=False))+'\n')
        f.write('MRE:'+str(mean_absolute_percentage_error(Y_true, y_pred))+'\n')
        f.write('R2:'+str(r2_score(Y_true, y_pred))+'\n')
        

