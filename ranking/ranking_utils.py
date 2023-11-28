import scipy
import numpy as np


def r_names(team, name_dict=None):
    if name_dict is None:
        name_dict = {'lasse_hansen_uzl': 'PDD-Net',
            'TAU': 'Lifshitz',
            'nifty': 'NiftyReg',
            'constance_fourcade_nantes': 'Epicure',
            'UCL': 'Multi-Brain',
            'niklas_gunnarsson_uppsala': 'Gunnarson',
            'driverliu_080523': 'DingkunLiu',
            'yihao2': 'YihaoLiu',
            'tm_esc': 'BinDuan',
            'junyu_chen1': 'JunyuChen',
            'nifty_2021': 'NiftyReg'}

    if team in name_dict.keys():
        return name_dict[team]
    else:
        return team


p_threshold = 0.05
def scores_better(task_metric,N):
    better = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            h,p = scipy.stats.ranksums(task_metric[i], task_metric[j])
            if((h>0)&(p<p_threshold)): #sign of h and p-value
                better[i,j] = 1
    scores_task = better.sum(0)
    return scores_task

def rankscore_avgtie(scores_int):#
    N = len(scores_int)
    rankscale = np.linspace(.1,1,N) #our definition
    rankavg = np.zeros((N,2))
    scorerank = np.zeros(N)
    #argsort with reverse index
    idx_ = np.argsort(scores_int)
    idx = np.zeros(N).astype('int32')
    idx[idx_] = np.arange(N)
    #averaging ties
    for i in range(N):
        rankavg[scores_int[i],0] += rankscale[idx[i]]
        rankavg[scores_int[i],1] += 1
    rankavg = rankavg[:,0]/np.maximum(rankavg[:,1],1e-6)
    for i in range(N):
        scorerank[i] = rankavg[scores_int[i]]
    return scorerank
scores = np.array([0, 2, 4, 3, 4, 0]).astype('int32')
scorerank = rankscore_avgtie(scores)

def greaters(scores):
    return np.sum(scores.reshape(1,-1)>scores.reshape(-1,1),0)


def rubustify(res_array, N, num_cases, num_labels, percentile = 0.3, initial_index = 0):
    '''
    res_array: array of results
    percentile: percentile of results to be used for robustification
    initial_index: index of initial results
    '''
    res0 = res_array[initial_index]  
    res = np.zeros((N, round((num_cases*num_labels)*percentile)))
    idx = np.argsort(res0.reshape(-1))[:res.shape[1]]
    for i in range(N):
        res[i] = res_array.reshape(N,-1)[i,idx]
    return res