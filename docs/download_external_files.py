from astropy.utils.data import download_file

version = 10059491
slug = "https://zenodo.org/record/{:d}/files/{:}"

fnames = [
    "logo_cube.noise.npz",
    "HD143006_continuum.npz",
    "logo_cube.tclean.fits",
]

for fname in fnames:
    url = slug.format(version, fname)
    download_file(
        url,
        cache=True,
        pkgname="mpol",
    )
