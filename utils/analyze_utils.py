import pandas as pd
import numpy as np

def getOveralStats(log_path, sofa_num=15, chose_best=False, algo_name='new_algo'):
    ''' Reads the end results of several algorithms and returns a pandas dataframe
    IMPORTANT!: this function assumes some hard naming constraints on the log file and should
    only be used with a proper formatting. This is not meant for public usage and shouldnt be
    commited to a public repository. For details or questions send an email to mtezcan@bu.edu.

    :inputs:
    :param log_path: (string) path to the log file
    :param sofa_num: (int, default=15) index of the last row which includes state-of-the-art result
    :param chose_best: (boolean, default=False) If True chose the best runs based on F-score
    :param algo_name: (string, default=new_algo) Name of the new algorithm
    '''

    if chose_best:
        raise Exception('choose_best=True is not implemented yet')
    # read the log file
    log_df = pd.read_csv(log_path, header=None, index_col=0).transpose()

    # Separate sofa and current algorithm
    sofa_df = log_df.iloc[:, :15]
    myalgo_df = log_df.iloc[:, 15:]


    myalgo_df = myalgo_df.astype(float)
    col_names = myalgo_df.columns.values.tolist()
    un_names = []
    med_names = []
    no_un_names = []

    for name in col_names:
        try:
            if name.split('_')[-2] == 'un':
                un_names.append(name)
            elif name.split('_')[-1] == 'med':
                med_names.append(name)
            else:
                no_un_names.append(name)
        except:
            break

    un_df = myalgo_df.drop(columns=no_un_names + med_names)
    med_df = myalgo_df.drop(columns=no_un_names + un_names)
    no_un_df = myalgo_df.drop(columns=un_names + med_names)

    med_result = med_df.transpose().replace(0, np.nan).mean(skipna=True)
    med_result[0] = 0
    nan_idx = ~med_result.apply(np.isnan)
    med_result = med_result.loc[nan_idx]
    med_result.name = '%s_med' %algo_name

    un_result = un_df.transpose().replace(0, np.nan).mean(skipna=True)
    un_result[0] = 0
    nan_idx = ~un_result.apply(np.isnan)
    un_result = un_result.loc[nan_idx]
    un_result.name = '%s_un' %algo_name

    no_un_result = no_un_df.transpose().replace(0, np.nan).mean(skipna=True)
    no_un_result[0] = 0
    nan_idx = ~no_un_result.apply(np.isnan)
    no_un_result = no_un_result.loc[nan_idx]
    no_un_result.name = '%s' %algo_name


    sofa_df = sofa_df.loc[nan_idx]
    sofa_df = sofa_df.astype(float, errors='ignore')

    myalgo_df = sofa_df.transpose().append(un_result).append(no_un_result).append(med_result)

    return myalgo_df
