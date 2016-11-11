import pandas as pd
import numpy as np
import seaborn
import matplotlib.pyplot as plt

def build_df():
    folder = '/home/matt/ISE5144_project/output/dea/'
    dmu = pd.read_csv(folder+'dmus.csv')
    eff = pd.read_csv(folder+'effscores.csv')
    df = dmu.merge(eff, on = 'DMU', how = 'inner')
    return df

def gen_plots(df):
    
    plt.subplot(2,2,1)
    df_1 = df[df['x'] == 1]
    plt.scatter(df_1['wind_mills'],df_1['panel_area'],
            c = df_1['storage'], cmap = 'autumn_r')
    plt.xlim([0,25])
    plt.ylim([0,2500])
    plt.xlabel('Number of Wind Mills')
    plt.ylabel('Panel Area (m2)')
    plt.title('Technically Efficient Configurations')
  
    plt.subplot(2,2,2)
    df_s = df[df['storage'] == 800]
    plt.scatter(df_s['total_prod'], df_s['x'])
    plt.xlabel('Energy Produced (kWh)')
    plt.ylabel('Efficiency Score')
    plt.title('Efficiency and Total Production')

 
    plt.subplot(2,2,3)
    plt.scatter(df_s['energy_purchased'], df_s['x'])
    plt.xlabel('Energy Purchased (kWh)')
    plt.ylabel('Efficiency Score')
    plt.title('Efficiency and Grid Purchases')

    
    plt.subplot(2,2,4)
    df_s = df[df['storage'] == 800]
    plt.scatter(df_s['reward_risk'], df_s['x'])
    plt.xlabel('Reward-Risk Coefficient')
    plt.ylabel('Efficiency Score')
    plt.title('Efficiency and Production Stability')

