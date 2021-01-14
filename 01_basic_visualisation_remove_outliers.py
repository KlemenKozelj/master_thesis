import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from os import listdir, makedirs
from scipy.stats import kstest
import missingno as msno
import seaborn as sns

outlier_fields = {
    'Temperature_Monte_Serra': [0],
    'Temperature_Orentano': [0],
    'Temperature_Ponte_a_Moriano': [0],
    'Volume_CSA': [0],
    'Volume_CSAL': [0]
}
    
def process_column(df, cn):
    print('   Processing column {}'.format(cn))
    
    histogram_bins=100
    histogram_range=(df[cn].min(), df[cn].max())
    scatter_X=(df['Date'].min(), df['Date'].max())
    scatter_Y=(df[cn].min(), df[cn].max())
    
    fig, axs = plt.subplots(4, figsize=(10, 13))
    fig.suptitle('Table: {}\nColumn: {}'.format(df.name, cn))
    
    axs[0].set_title('range={}-{} bins={}'.format(histogram_range[0], histogram_range[1], histogram_bins))
    n, bins, _ = axs[0].hist(df[cn], range=histogram_range, bins=histogram_bins)
    
    axs[1].set_xlim(scatter_X)
    axs[1].set_ylim(scatter_Y)
    axs[1].scatter(x=df['Date'], y=df[cn], marker='o', s=1, alpha=1)

    percentile = 2.5
    if cn in outlier_fields:
        df[cn] = df[cn].replace(outlier_fields[cn], float('NaN'))
    df.loc[df[cn] > np.percentile(df[cn].dropna(), 100-percentile), cn] = float('NaN')
    df.loc[df[cn] < np.percentile(df[cn].dropna(), percentile), cn] = float('NaN')
    
    axs[2].set_ylim((0, axs[0].get_ylim()[1]))
    axs[2].set_title('kept percentile {} < x < {}'.format(percentile, 100-percentile))
    axs[2].hist(df[cn], range=histogram_range, bins=histogram_bins)

    axs[3].set_xlim(scatter_X)
    axs[3].set_ylim(scatter_Y)
    axs[3].scatter(x=df['Date'], y=df[cn], marker='o', s=1, alpha=1)

    plt.savefig('{}_column_{}.png'.format(df.output_file, cn))
    #plt.show()
    plt.close()

def basic_visulisation(df, pr):
    msno.matrix(df).figure.savefig('{}_{}_notnan_matrix.png'.format(df.output_file, pr), bbox_inches = 'tight')
    msno.bar(df).figure.savefig('{}_{}_notnan_bar.png'.format(df.output_file, pr), bbox_inches = 'tight')
    msno.heatmap(df, cmap='coolwarm').figure.savefig('{}_{}_notnan_heatmap.png'.format(df.output_file, pr), bbox_inches = 'tight')

    corr = df.corr(method='pearson')
    sns.heatmap(corr, mask=np.triu(np.ones_like(corr, dtype=bool)), xticklabels=corr.columns, yticklabels=corr.columns, vmin=-1, vmax=1, cmap='coolwarm').figure.savefig('{}_{}_corr.png'.format(df.output_file, pr), bbox_inches = 'tight')

def process_file(input_directory, filename):
    df = pd.read_csv('{}/{}'.format(input_directory, filename), sep=',', header=0)
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

    df.name=filename.split('.')[0]
    df.output_directory='{}/{}'.format('./results/01_basic_visualisation_remove_outliers', df.name)
    makedirs(df.output_directory, exist_ok=True)
    df.output_file='{}/{}'.format(df.output_directory, df.name)

    basic_visulisation(df, 'raw')
    for cl in df.columns[1:]:
        process_column(df, cl)
    basic_visulisation(df, 'prcs')
    df.to_csv('./data/01_removed_outliers/{}.csv'.format(df.name), header=True, index=True)

input_directory = './data/00_original'
for filename in listdir(input_directory):
    if filename.endswith('.csv'):
        print('Processing filename ' + filename)
        process_file(input_directory, filename)