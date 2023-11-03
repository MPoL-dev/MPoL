from astropy.utils.data import download_file
from mpol.__init__ import zenodo_record

slug = "https://zenodo.org/record/{:d}/files/{:}"

fnames = [
    "logo_cube.noise.npz",
    "HD143006_continuum.npz",
    "logo_cube.tclean.fits",
]

for fname in fnames:
    url = slug.format(zenodo_record, fname)
    download_file(
        url,
        cache=True,
        pkgname="mpol",
    )
