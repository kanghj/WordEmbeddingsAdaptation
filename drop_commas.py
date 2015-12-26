"""
Makes txt file out of csv files
"""

import glob

if __name__ == "__main__":
    csv_filenames = glob.glob('./testfiles/*.csv')

    for csv_filename in csv_filenames:
        with open(csv_filename) as csv_file, \
             open('./testtxt/' + csv_filename.strip('.csv') + '.txt', 'w+') as output_file:
             for line in csv_file:
                  output_file.write(line.replace(',', ' '))
        print 'done with ', csv_filename
            
