import pandas as pd
from os import listdir
import missingno as msno

input_directory = './data/original'
output_directory = './results/original_data_analysis'

def analyse_file(input_directory, filename):
    df = pd.read_csv('{}/{}'.format(input_directory, filename), sep=',', header=0)

    output_file = '{}/{}'.format(output_directory, filename.split('.')[0])

    msno.matrix(df).figure.savefig('{}_notnan_matrix.png'.format(output_file), bbox_inches = 'tight')
    msno.bar(df).figure.savefig('{}_notnan_bar.png'.format(output_file), bbox_inches = 'tight')
    msno.heatmap(df).figure.savefig('{}_notnan_heatmap.png'.format(output_file), bbox_inches = 'tight')

    text_file = open('{}_metadata.txt'.format(output_file), 'w')

    text = 'Generate descriptive statistics, excluding NaN values\n\n'
    text += df.describe().to_string() + '\n\n'
    text += 'Count non-NA cells for each column\n\n'
    text += df.count(axis='rows').to_string()
    #text += df.count(axis='columns').to_string() + '\n\n'

    text_file.write(text)
    text_file.close()

for filename in listdir(input_directory):
    if filename.endswith('.csv'):
        analyse_file(input_directory, filename)