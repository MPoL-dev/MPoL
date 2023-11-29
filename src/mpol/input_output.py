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


    def get_extent(self, header):
        """Get extent (in RA and Dec, units of [arcsec]) of image"""
        
        # get the coordinate labels
        nx = header["NAXIS1"]
        ny = header["NAXIS2"]

        assert (
            nx % 2 == 0 and ny % 2 == 0
        ), f"Image dimensions x {nx} and y {ny} must be even."

        # RA coordinates
        CDELT1 = 3600 * header["CDELT1"]  # arcsec (converted from decimal deg)
        # CRPIX1 = header["CRPIX1"] - 1.0  # Now indexed from 0

        # DEC coordinates
        CDELT2 = 3600 * header["CDELT2"]  # arcsec
        # CRPIX2 = header["CRPIX2"] - 1.0  # Now indexed from 0

        RA = (np.arange(nx) - nx / 2) * CDELT1  # [arcsec]
        DEC = (np.arange(ny) - ny / 2) * CDELT2  # [arcsec]

        # extent needs to include extra half-pixels.
        # RA, DEC are pixel centers

        ext = (
            RA[0] - CDELT1 / 2,
            RA[-1] + CDELT1 / 2,
            DEC[0] - CDELT2 / 2,
            DEC[-1] + CDELT2 / 2,
        )  # [arcsec]

        return RA, DEC, ext


    def get_beam(self, hdu_list, header):
        """Get the major and minor widths [arcsec], and position angle, of a 
        clean beam"""

        if header.get("CASAMBM") is not None:
            # Get the beam info from average of record array
            data2 = hdu_list[1].data
            BMAJ = np.median(data2["BMAJ"])
            BMIN = np.median(data2["BMIN"])
            BPA = np.median(data2["BPA"])
        else:
            # Get the beam info from the header, like normal
            BMAJ = 3600 * header["BMAJ"]
            BMIN = 3600 * header["BMIN"]
            BPA = header["BPA"]

        return BMAJ, BMIN, BPA


    def get_image(self, beam=True):
        """Load a .fits image and return as a numpy array. Also return image 
        extent and optionally (`beam`) the clean beam dimensions"""

        hdu_list = fits.open(self._fits_file)
        hdu = hdu_list[0]
        
        if len(hdu.data.shape) in [3, 4]:
            image = hdu.data[self._channel]  # first channel
        else:
            image = hdu.data

        image *= 1e3

        if len(image.shape) == 3:
            image = np.squeeze(image)

        header = hdu.header

        RA, DEC, ext = self.get_extent(hdu.header)

        if beam:
            return image, ext, self.get_beam(hdu_list, header)
        else:
            return image, ext
        