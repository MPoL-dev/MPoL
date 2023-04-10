import matplotlib.pyplot as plt
import torch
from pytest import approx

from mpol import primary_beam, images, utils
from mpol.constants import *

def test_no_dish_correction(coords, unit_cube):
    # Tests layer when no PB correction is applied (passthrough layer)
    pblayer = primary_beam.PrimaryBeamCube(coords=coords)
    out_cube = pblayer(unit_cube)
    
    assert torch.equal(unit_cube, out_cube)
    
    
    