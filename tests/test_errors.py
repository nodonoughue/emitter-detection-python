import numpy as np

from ewgeo.utils import errors

def test_cep50():

    res = errors.compute_cep50(np.array([[1, 0],[0, 1]]))
    assert res == 1.18

def test_cep50_3d():

    res = errors.compute_cep50(np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]]))
    assert res == 1.18

def test_rmse_scaling():
    inputs = [0.01, .1, .2, .3, .4, .5, .6, .7, .8, .9, .99]
    outputs = [0.012533469508069276,
               0.12566134685507416,
               0.2533471031357997,
               0.38532046640756773,
               0.5244005127080407,
               0.6744897501960817,
               0.8416212335729143,
               1.0364333894937898,
               1.2815515655446004,
               1.6448536269514722,
               2.5758293035489004]
    assert all(errors.compute_rmse_scaling(x)==y for x, y in zip(inputs, outputs))

def test_rmse_confidence_interval():
    inputs = np.arange(20) - 10
    outputs = [-1.0, -1.0,
               -0.9999999999999987,
               -0.9999999999974403,
               -0.9999999980268246,
               -0.9999994266968562,
               -0.9999366575163338,
               -0.9973002039367398,
               -0.9544997361036416,
               -0.6826894921370859,
               0.0,
               0.6826894921370859,
               0.9544997361036416,
               0.9973002039367398,
               0.9999366575163338,
               0.9999994266968562,
               0.9999999980268246,
               0.9999999999974403,
               0.9999999999999987,
               1.0]
    assert all(errors.compute_rmse_confidence_interval(x) == y for x, y in zip(inputs, outputs))

# ToDo: unit test for draw_cep50 and draw_error_ellipse