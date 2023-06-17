# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 16:11:35 2023

@author: benja
"""

import pandas as pd
import trn as tr
import trn1 as tr1
import tst as ts
import matplotlib.pyplot as plt

def main(runs = 5):
    
    cost_df = pd.DataFrame()
    fsco_df = pd.DataFrame()
    
    for i in range(runs):
        print('Corriendo run_',i+1)
        tr.main()
        ts.main()
        
        cost_df_i = pd.read_csv('costo.csv', converters={'COLUMN_NAME': pd.eval}, header=None)
        fsco_df_i = pd.read_csv('fscores.csv', converters={'COLUMN_NAME': pd.eval}, header=None)
        
        cost_df.insert(i, 'run_'+str(i), cost_df_i)
        fsco_df.insert(i, 'run_'+str(i), fsco_df_i)
    
    fsco_df = fsco_df.multiply(100)
    fsco_df = fsco_df.astype('int')
    
    cost_df.to_csv('costo_runs.csv',index=False, header = False )
    fsco_df.to_csv('fscores_runs.csv',index=False, header = False )
    
    plt.plot(cost_df)
    plt.title('Runs_cost')
    plt.show()
    
if __name__ == '__main__':
	 main()
