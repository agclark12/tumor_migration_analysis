# tumor_migration_analysis

This package contains several python scripts and ImageJ/FIJI macros
that can aid in the analysis of collective migration patterns.

All python scripts were written and tested in python 3.7.x

## Requirements

It is recommended to download the [Anaconda Distribution](https://www.anaconda.com/products/individual) of python3 to use these scripts.
The Anaconda distribution will contain most of the necessary packages you need.

To download the additional required packages, in a terminal window, type `pip install` or `conda install` followed by the name of the package.
Additional packages not included in Anaconda python include:

- shapely~=1.8.1
- openpiv~=0.23.8
- astropy~=4.3.1

Alternatively, and what is probably a better solution, is to start a virtual environment for this package. To do so, follow these steps in a terminal window:

1. Clone or download this repository
2. Using a terminal, navigate to the directory where this README is stored
3. Create a virtual environment using `python -m venv venv`
4. Activate your virtual environment using `source venv/bin/activate` (for Windows users `vev\Scripts\activate`)
5. Download the required packages using `python -m pip install -r requirements.txt`

Once you have python and the necessary packages installed, you can run any of the included scripts using python `name_of_script.py`
or by setting up the package in your favorite IDE.
(Be sure to configure the interpreter to use your virtual environment if you have set this up!)

The ```.ijm``` files are macros for ImageJ/FIJI and can be run directly in the FIJI script editor.

For more information on how to incorporate these scripts into your analysis pipline for collective tumor migration, please see our accompanying article:

Staneva and Clark (2022) Analysis of Collective Migration Patterns within Tumors. In: Methods in Molecular Biology - Cell Migration in Three Dimensions, C. Margadant, ed. (Springer Nature, London, UK), pp. XX-XX.

If you found these scripts to be useful, please feel free to cite our publication.

## Known Issues

1. One some Windows machines you may get an error when running `piv_extract_vectors.py` that looks something like:

```
ImportError: DLL load failed: The specified module could not be found.
```

This is caused by an issue in importing `astropy` and can be solved by copying the following files from `...\anaconda3\Library\bin` and pasting them in `...\anaconda3\DLLs`:

```
libcrypto-1_1-x64.dll
libssl-1_1-x64.dll 
```

Details and credits [here](https://stackoverflow.com/a/60342954/855617)