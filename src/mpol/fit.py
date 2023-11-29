# Much of the syntax in this file closely follows that in `frank` 
# (see https://github.com/discsim/frank/blob/master/frank/fit.py).

import os
import json
import argparse
import logging

import numpy as np

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


def parse_parameters(*args):
    """
    Read in a .json parameter file to set the fit parameters. 

    Parameters
    ----------
    parameter_filename : string, default `default_parameters.json`
        Parameter file (.json; see mpol.fit.helper)
    data_filename : string
        Data file with visibilities to be fit (.txt, .npy, or .npz).
        For .txt, the column format should be: 
        u [klambda] v [klambda] Re(V) + 1j * Im(V) [Jy] Weight [Jy^-2] 
        # TODO: confirm format and update parameter_descriptions

    Returns
    -------
    config : dict
        Dictionary containing parameters the modeling pipeline uses
    param_path : string
        Path to .json parameter file in which used model parameters are saved
    """

    default_param_file = os.path.join(mpol_path, 'default_parameters.json')

    parser = argparse.ArgumentParser("Run an MPol fit, by default using"
                                     " parameters in default_parameters.json")
    parser.add_argument("-p", "--parameter_filename",
                        default=default_param_file, type=str,
                        help="Parameter file (.json; see mpol.fit.helper)")
    parser.add_argument("-data", "--data_filename", default=None, type=str,
                        help="Data file with visibilities to be fit. See"
                             " mpol.io.load_data") # TODO: point to correct load_data routine location
    parser.add_argument("-desc", "--print_parameter_description", default=None,
                        action="store_true",
                        help="Print the full description of all fit parameters")

    args = parser.parse_args(*args)

    if args.print_parameter_description:
        helper()
        exit()

    config = json.load(open(args.parameter_filename, 'r'))

    if args.data_filename:
        config['input_output']['data_filename'] = args.data_filename

    if ('data_filename' not in config['input_output'] or
            not config['input_output']['data_filename']):
        raise ValueError("data_filename isn't specified."
                         " Set it in the parameter file or run MPoL with"
                         " python -m mpol.fit -data <data_filename>")

    data_path = config['input_output']['data_filename']
    if not config['input_output']['save_dir']:
        # If not specified, use the data file directory as the save directory
        config['input_output']['save_dir'] = os.path.dirname(data_path)

    # Add a save prefix to the .json parameter file for later use
    config['input_output']['save_prefix'] = save_prefix =  \
        os.path.join(config['input_output']['save_dir'],
                     os.path.splitext(os.path.basename(data_path))[0])

    # enable logger, printing output and writing to file
    log_path = save_prefix + '_mpol_fit.log'
    mpol.enable_logging(log_path) 

    logging.info('\nRunning MPoL on'
                 ' {}'.format(config['input_output']['data_filename']))

    # TODO: add par sanity checks

    param_path = save_prefix + '_mpol_used_pars.json'

    logging.info(
        '  Saving parameters used to {}'.format(param_path))
    with open(param_path, 'w') as f:
        json.dump(config, f, indent=4)

    return config, param_path


def load_data(config): # TODO

def modify_data(config): # TODO

def train_test_crossval(config): # TODO 

def output_results(config): # TODO

def main(*args):
    """Run the full MPoL pipeline to fit a dataset

    Parameters
    ----------
    *args : strings
        Simulates the command line arguments
    """

    config, param_path = parse_parameters(*args)

    # TODO: add pipeline 

    logging.info('  Updating {} with final parameters used'
                    ''.format(param_path))
    with open(param_path, 'w') as f:
        json.dump(config, f, indent=4)

    logging.info("MPoL MCoMplete!\n")


if __name__ == "__main__":
    main()
