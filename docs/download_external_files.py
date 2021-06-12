from astropy.utils.data import download_file

version = 4930016
slug = "https://zenodo.org/record/{:d}/files/{:}"

fnames = [
    "logo_cube.noise.npz",
    "HD143006_continuum.npz",
]

for fname in fnames:
    url = slug.format(version, fname)
    download_file(
        url,
        cache=True,
        pkgname="mpol",
    )
