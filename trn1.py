# Training DL via mAdam

import pandas as pd
import numpy as np
import utility as ut


# Training miniBatch
def train_sft_batch(X, Y, W, V, S, t, Param):
    costo = []
    M = Param[2]
    numBatch = np.int16(np.floor(X.shape[1]/M))
    for i in range(numBatch):
        Idx = ut.get_Idx_n(i, M)
        xe, ye = X[:, slice(*Idx)], Y[:, slice(*Idx)]

        z = np.dot(W[0], xe)
        a = ut.softmax(z)

        gW, Cost = ut.gradW_softmax(xe, ye, a)

        W, V, S = ut.updW_madam(W, V, S, [gW], t, u=Param[1])
        t+=1
        costo.append(Cost)

    return W, V, S, costo,t

# Softmax's training via mAdam


def train_softmax(X, Y, Param):
    W = [ut.iniW(Y.shape[0], X.shape[0])]
    V, S = [np.zeros((Y.shape[0], X.shape[0]))], [np.zeros((Y.shape[0], X.shape[0]))]
    Cost = []
    t=1
    idx = np.random.permutation(X.shape[1])
    xe, ye = X[:, idx], Y[:, idx]
    for Iter in range(1, Param[0]+1):
        
        W, V, S, c, t = train_sft_batch(xe, ye, W, V, S,t, Param)

        Cost.append(np.mean(c))

        if Iter % 10 == 0:
            print('\tIterar-SoftMax: ', Iter, ' Cost: ', Cost[Iter-1])

    return W, Cost

# Training by using miniBatch


def train_dae_batch(X, W, V, S,t, Param):

    numBatch = np.int16(np.floor(X.shape[1]/Param[6]))
    cost = []

    for i in range(numBatch):
        Idx = ut.get_Idx_n(i, Param[6])
        xe = X[:, slice(*Idx)]

        Act = ut.dae_forward(xe, W, Param)

        gW, Cost = ut.gradW(Act, W, Param)

        W, V, S = ut.updW_madam(W, V, S, gW, t, u=Param[7])
        t+=1
        cost.append(Cost)
    return W, V, S, cost,t

# DAE's Training


def train_dae(x, Param):
    # W,V,S = ut.iniW()

    Param_ = Param[8:]

    W, V, S = ut.iniWs(x.shape[0], Param_)
    t=1
    Cost = []
    xe = x[:, np.random.permutation(x.shape[1])]
    for Iter in range(1, Param[5]+1):
        
        W, V, S, c , t = train_dae_batch(xe, W, V, S, t, Param)

        Cost.append(np.mean(c))
        if Iter % 10 == 0:
            print('\tIterar-AE: ', Iter, ' Cost: ', Cost[Iter-1])
    
    N = len(W)//2
    W = W[:N]
    
    Act = ut.dae_forward(x, W, Param)
    X = Act[-1][-1]
    
    return W, X


# load Data for Training
def load_data_trn(ruta_archivo='train.csv'):
    df = pd.read_csv(ruta_archivo, converters={'COLUMN_NAME': pd.eval})

    X = df.filter(regex='x_')
    Y = df.filter(regex='y_')
    return np.asarray(X).T, np.asarray(Y).T


# Beginning ...
def main():
    p_dae, p_sft = ut.load_cnf_dae()
    xe, ye = load_data_trn()   
    W, Xr = train_dae(xe, p_dae)
    Ws, cost = train_softmax(Xr, ye, p_sft)
    ut.save_w_dl(W, Ws, cost)


if __name__ == '__main__':
    main()
