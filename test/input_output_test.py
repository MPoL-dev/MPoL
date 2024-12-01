
from astropy.utils.data import download_file
from mpol.input_output import ProcessFitsImage


# def test_ProcessFitsImage():
#     # get a .fits file produced with casa
#     fname = download_file(
#         "https://zenodo.org/record/4711811/files/logo_cube.tclean.fits",
#         cache=True,
#         show_progress=True,
#         pkgname="mpol",
#     )

#     fits_image = ProcessFitsImage(fname)
#     clean_im, clean_im_ext, clean_beam = fits_image.get_image(beam=True)
