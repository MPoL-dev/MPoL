# get the location of the README.md file 
import os
# path = os.path.dirname(__file__)

path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'README.md'))

with open(path) as version_file:
    for line in version_file:
        pass
    version = line.strip()

__version__ = version