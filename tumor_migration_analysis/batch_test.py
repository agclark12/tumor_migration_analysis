#!/Users/clark/anaconda3/bin/python

"""


"""

import os

from piv_extract_vectors import extract_vectors
from piv_plot_vectors import plot_vectors
from piv_analyze_vectors import analyze_vectors


def main():

    data_dir = '/home/data'

    for filename in os.listdir(data_dir):
        if '.tif' in data_dir:

            stk_path = data_dir + '/' + filename
            time_int = 2
            px_size = 0.5
            window_len  = 10


            extract_vectors(stk_path,time_int=time_int,px_size=px_size,window_length=window_len)
            plot_vectors(stk_path,px_size=px_size,scale_factor=0.4,scale_length=0.1)
            analyze_vectors(stk_path)


if __name__ == "__main__":
    main()
