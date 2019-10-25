import glob, os
import pandas as pd
pd.set_option('display.max_colwidth', -1) # displays content from given folder


#function to rename given files
def rename(dir, pathAndFilename, pattern, titlePatteren):
    os.remane(pathAndFilename, os.path.join(dir, titlePatteren)) # methdoe to do the rename

# search for csv files in working folder
path = os.path.expanduser("/PycharmProjects/Jarvis/*.csv") #pass folder name here

#iterate and remane them one by one with the number of the iteration
for i, fname in enumerate(glob.glob(path)): # change file name to add numbee
    rename(os.path.expanduser('~/PycharmProjects/Jarvis/'), fname, r'*.csv', r'test{}.csv'.format(i))# change file name without changing extention e.g ".csv"

#Change spearator for CSV file, name file location here
df1 = pd.read_csv('~/PycharmProjects/Jarvis/opel_corsa_01.csv', sep=";")
df2 = pd.read_csv('~/PycharmProjects/Jarvis/opel_corsa_02.csv', sep=";")
df3 = pd.read_csv('~/PycharmProjects/Jarvis/peugeot_207_01.csv', sep=";")
df4 = pd.read_csv('~/PycharmProjects/Jarvis/peugeot_207_02.csv', sep=";")

frames = [df1, df2, df3, df4]

#concatente multiple data csv files
all = pd.concat(frames)

#print(df1.shape)
#print(df2.shape)
#print(df3.shape)
#print(df4.shape)
print(all.shape)