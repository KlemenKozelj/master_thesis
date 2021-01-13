import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from os import listdir, makedirs
from scipy import stats
import missingno as msno
import seaborn as sns

input_directory = './data/original'
output_directory = './results/original_data_visualisation'

def analyse_file(input_directory, filename):
    df = pd.read_csv('{}/{}'.format(input_directory, filename), sep=',', header=0)
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

    name=filename.split('.')[0]
    output_filename_directory = '{}/{}'.format(output_directory, name)
    makedirs(output_filename_directory, exist_ok=True)
    output_file = '{}/{}'.format(output_filename_directory, name)

    msno.matrix(df).figure.savefig('{}_notnan_matrix.png'.format(output_file), bbox_inches = 'tight')
    msno.bar(df).figure.savefig('{}_notnan_bar.png'.format(output_file), bbox_inches = 'tight')
    msno.heatmap(df, cmap='coolwarm').figure.savefig('{}_notnan_heatmap.png'.format(output_file), bbox_inches = 'tight')

    corr = df.corr(method='pearson')
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, xticklabels=corr.columns, yticklabels=corr.columns, vmin=-1, vmax=1, cmap='coolwarm').figure.savefig('{}_correlations.png'.format(output_file), bbox_inches = 'tight')

    text_file = open('{}_metadata.txt'.format(output_file), 'w')
    text = 'Generate descriptive statistics, excluding NaN values\n\n'
    text += df.describe(include='all', datetime_is_numeric=True).to_string() + '\n\n'
    text += 'Count non-NA cells for each column\n\n'
    text += df.count(axis='rows').to_string()
    text_file.write(text)
    text_file.close()

    for c in df.columns[1:]:
        print('   Processing column ' + c)
        bins=100
        ran=(df[c].min(), df[c].max())
        
        fig, axs = plt.subplots(2, figsize=(10, 10))

        fig.suptitle('{}: {}'.format(name, c))

        axs[0].set_title('distribution range={}-{} bins={}'.format(ran[0], ran[1], bins))
        axs[0].hist(df[c], range=ran, bins=bins, histtype='bar', align='mid')

        axs[1].set_title('chronological')
        axs[1].scatter(x=df['Date'], y=df[c], marker='o', s=1, alpha=0.7)

        plt.savefig('{}_column_{}.png'.format(output_file, c))
        plt.close()

for filename in listdir(input_directory):
    if filename.endswith('.csv'):
        print('Processing filename ' + filename)
        analyse_file(input_directory, filename)