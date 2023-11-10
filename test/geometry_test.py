import torch
from pytest import approx
import numpy as np 

from mpol import geometry

def test_rotate_points():
    '''
    Test rotation from flat 2D frame to observer frame and back
    '''
    xs = torch.tensor([0.0, 1.0, 2.0])
    ys = torch.tensor([1.0, -1.0, 2.0])

    omega = 35. * np.pi/180
    incl = 30. * np.pi/180
    Omega = 210. * np.pi/180

    X, Y = geometry.flat_to_observer(xs, ys, omega=omega, incl=incl, Omega=Omega)

    xs_back, ys_back = geometry.observer_to_flat(X, Y, omega=omega, incl=incl, Omega=Omega)


    print("original", xs, ys)
    print("Observer", X, Y)
    print("return", xs_back, ys_back)

    assert xs == approx(xs_back, abs=1e-6)
    assert ys == approx(ys_back, abs=1e-6)


def test_rotate_coords(coords):
    
    omega = 35. * np.pi/180
    incl = 30. * np.pi/180
    Omega = 210. * np.pi/180

    x, y = geometry.observer_to_flat(coords.sky_x_centers_2D, coords.sky_y_centers_2D, omega=omega, incl=incl, Omega=Omega)
    
    print(x, y)

    
    