# Much of the syntax in this file closely follows that in `frank` 
# (see https://github.com/discsim/frank/blob/master/frank/fit.py).

import os
import json

import numpy as np

# import logging # TODO: use?

import mpol
# from mpol import # TODO

mpol_path = os.path.dirname(mpol.__file__)

def get_default_parameter_file():
    """Get the path to the default parameter file"""
    return os.path.join(mpol_path, 'default_parameters.json')


def load_default_parameters():
    """Load the default parameters"""
    return json.load(open(get_default_parameter_file(), 'r'))


def get_parameter_descriptions():
    """Get the description for parameters"""
    with open(os.path.join(mpol_path, 'parameter_descriptions.json')) as f:
        param_descrip = json.load(f)
    return param_descrip


def helper():
    param_descrip = get_parameter_descriptions()

    print("""
         Forward model a 2D image with MPoL from the terminal with 
         `python -m mpol.fit`. A .json parameter file is required;
         the default is default_parameters.json and is
         of the form:\n\n {}""".format(json.dumps(param_descrip, indent=4)))


