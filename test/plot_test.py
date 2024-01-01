import numpy as np

from astropy.utils.data import download_file

from mpol import precomposed
# from mpol.plot import image_comparison_fig

# def test_image_comparison_fig(coords, tmp_path):
#     # generate an image comparison figure

#     model = precomposed.SimpleNet(coords=coords, nchan=1)
#     model()

#     # just interested in whether the tested functionality runs
#     u = v = np.repeat(1e3, 1000)
#     V = weights = np.ones_like(u)

#     # .fits file to act as clean image
#     fname = download_file(
#         "https://zenodo.org/record/4711811/files/logo_cube.tclean.fits",
#         cache=True,
#         show_progress=True,
#         pkgname="mpol",
#     )

#     image_comparison_fig(model, u, v, V, weights, robust=0.5, 
#                             clean_fits=fname,
#                             share_cscale=False, 
#                             xzoom=[-2, 2], yzoom=[-2, 2],
#                             title="test",
#                             save_prefix=None,                            
#                             )
