import os

for root, directories, files in os.walk('./Nasrabadi_MMSys_19/dataset/Traces/'):
    print('root: ', root)
    print('directories: ', directories)
    print('files: ', files)
    print('-----------------')
    