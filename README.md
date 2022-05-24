# tumor_migration_analysis

This package contains several python scripts and ImageJ/FIJI macros
that can aid in the analysis of collective migration patterns.

All python scripts were written and tested in python 3.7.x

It is recommended to download the [Anaconda Distribution](https://www.anaconda.com/products/individual) of python3 to use these scripts.
The Anaconda distribution will contain most of the necessary packages you need.

To download the additional required packages, in a terminal window, type ```pip install``` or ```conda install``` followed by the name of the package.
Additional packages not included in Anaconda python include:

- shapely
- openpiv 
- astropy

Alternatively, and what is probably a better solution, is to start a virtual environment for this package. To do so, follow these steps in a terminal window:

1. Clone or download this repository
2. Navigate to the directory where this README is stored
3. Create a virtual environment using<br>```python3 -m venv venv```
4. Activate your virtual environment using<br>```source venv/bin/activate```
5. Download the required packages by typing<br>```pip install -r requirements.txt```

Once you have python and the necessary packages installed, you can run any of the included scripts using ```python name_of_script.py```
or by setting up the package in your favorite IDE.
(Be sure to configure the interpreter to use your virtual environment if you have set this up!)

The ```.ijm``` files are macros for ImageJ/FIJI and can be run directly in the FIJI script editor.

For more information on how to incorporate these scripts into your analysis pipline for collective tumor migration, please see our accompanying article:

Staneva and Clark (2022) Analysis of Collective Migration Patterns within Tumors. In: Methods in Molecular Biology - Cell Migration in Three Dimensions, C. Margadant, ed. (Springer Nature, London, UK), pp. XX-XX.

If you found these scripts to be useful, please feel free to cite our publication.