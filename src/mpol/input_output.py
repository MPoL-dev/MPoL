import numpy as np 

from astropy.io import fits

class ProcessFitsImage:
    """
    Utilities for loading and retrieving metrics of a .fits image

    Parameters
    ----------
    filename : str
        Path to the .fits file
    channel : int, default=0
        Channel of the image to access
    """

    def __init__(self, filename, channel=0):
        self._fits_file = filename
        self._channel = channel

