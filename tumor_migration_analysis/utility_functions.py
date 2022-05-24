#!/Users/clark/anaconda3/bin/python

"""


"""

import os
import re

def make_dir(path):
    """A convenience function for making a new directory

    Parameters
    ----------
    path : string
        the path of the directory to be created

    """

    if not os.path.isdir(path):
        os.mkdir(path)

def save_data_array(array, save_path):
    """Saves a rectangular data array to a text file with even rows and columns

    Parameters
    ----------
    array : list of lists (2D; must be rectangular)
        the data array to be saved

    """

    if not all(len(_) == len(array[0]) for _ in array): #makes sure that the data array is rectangular

        raise ValueError("The data array is not rectangular")

    else: #otherwise, continue with saving the array

        # finds column width
        column_width_list = []
        for column in zip(*array):
            column = map(str,column)
            column_width = max(len(x) for x in column) + 2
            column_width_list.append(column_width)

        # saves array to file
        ofile = open(save_path,'w')
        for i in range(len(array)):
            for j in range(len(array[i])):
                element = str(array[i][j]).ljust(column_width_list[j])
                ofile.write(element + '  ')
            ofile.write('\n')
        ofile.close()


def write_csv(array,path):
    """Writes a csv file from a 2D array

    Parameters
    ----------
    array : 2D list of lists
        the data you would like to save
    path : string
        path where the .csv file should be saved
    """


    ofile = open(path,'w')
    for i in range(len(array)):
        line = ','.join([str(_) for _ in array[i]]) + '\n'
        ofile.write(line)
    ofile.close()

def read_file(filename, delimiter=None, startline=0):
    """Reads a text file into a list of lists

    Parameters
    ----------
    filename : string
        the file path to open
    delimiter : string (optional)
        the delimiter separating the column data (default to None for all white space)
    startline : int (optional)
        the line to stat reading the data (default to 0)

    Returns
    -------
    data_list : list of lists (2D)
        the data array

    """

    data_list = []
    ifile = open(filename,'rU')

    for line in ifile:

        if delimiter:
            data = line.split(delimiter)
        else:
            data = line.split()
        data_list.append(data)

    ifile.close()
    return data_list[startline:]

def get_dict_list(data_array):
    """Returns a list of dictionaries based on the column headers from a list of lists
    (the 0th line is the column headers)

    Parameters
    ----------
    data_array : list of lists (2D)
        the data array

    Returns
    -------
    dict_list : dictionary
        a list of dictionaries containing the daa

    """

    key_list = data_array[0]
    dict_list = []
    for index, line in enumerate(data_array[1:]):
        params = {}
        for i in range(len(key_list)):
            # try:
            #     params[key_list[i]] = float(line[i])
            # except ValueError:
                params[key_list[i]] = line[i]
        dict_list.append(params)

    return dict_list

def natural_sort(ell):
    """Returns a naturally sorted list (taking numbers into account)

    Parameters
    ----------
    ell : list
        the list to be sorted

    Returns
    -------
    ell : list
        the sorted list

    """

    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(ell, key = alphanum_key)
