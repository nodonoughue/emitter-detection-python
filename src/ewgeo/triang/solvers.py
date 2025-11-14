from itertools import combinations
import math
import numpy as np
import numpy.typing as npt

from ewgeo.utils.geo import find_intersect


def angle_bisector(x_sensor: npt.NDArray[np.float64], zeta: npt.NDArray[np.float64])-> npt.NDArray[np.float64]:
    """
    Compute the center via intersection of angle bisectors for  
    3 or more LOBs given by sensor positions xi and angle 
    of arrival measurements zeta.

    If more than 3 measurements are provided, this method repeats on each set
    of 3 measurements, and then takes the average of the solutions.
    
    Ported from MATLAB code.
    
    Nicholas O'Donoughue
    22 February 2021
    
    :param x_sensor: N x 2 matrix of sensor positions
    :param zeta: N x 1 vector of AOA measurements (radians)
    :return x_est: 2 x 1 vector of estimated source position
    """

    # Parse Inputs
    sensor_sets, num_sets = _parse_sensor_triplets(x_sensor)

    # Loop over sets -- adding each estimate to x_est
    x_est = np.zeros((2, ))

    for sensor_set in sensor_sets:
        # Find vertices
        v0, v1, v2 = _find_vertices(x_sensor[:, np.asarray(sensor_set)], zeta[np.asarray(sensor_set)])

        # Find angle bisectors
        th_fwd0 = np.arctan2(v1[1]-v0[1], v1[0]-v0[0])
        th_fwd1 = np.arctan2(v2[1]-v1[1], v2[0]-v1[0])

        th_back0 = np.arctan2(v2[1]-v0[1], v2[0]-v0[0])
        th_back1 = np.arctan2(v0[1]-v1[1], v0[0]-v1[0])

        th_bi0 = .5*(th_fwd0 + th_back0)
        th_bi1 = .5*(th_fwd1 + th_back1)

        # Add pi if the angle is flipped (pointing out)
        if th_bi0 < th_fwd0:
            th_bi0 += np.pi

        # Add pi if the angle is flipped (pointing out)
        if th_bi1 < th_fwd1:
            th_bi1 += np.pi

        # Find the intersection
        this_cntr = find_intersect(v0, th_bi0, v1, th_bi1)

        # Accumulate the estimates, we'll divide by num_sets later to make it an average.
        x_est += this_cntr

    return x_est/num_sets


def centroid(x_sensor: npt.NDArray[np.float64], zeta: npt.NDArray[np.float64])-> npt.NDArray[np.float64]:
    """
    Compute the centroid of the intersection of 3 or more LOBs given by
    sensor positions x_source and angle of arrival measurements zeta.

    If more than 3 measurements are provided, this method repeats on each set
    of 3 measurements, and then takes the average of the solutions.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    22 February 2021

    :param x_sensor: 2 x N matrix of sensor positions
    :param zeta: N x 1 vector of AOA measurements (radians)
    :return x_est: 2 x 1 vector of estimated source position
    """

    # Parse Inputs
    sensor_sets, num_sets = _parse_sensor_triplets(x_sensor)

    # Loop over sets -- adding each estimate to x_est
    x_est = np.zeros(shape=(2, ))

    for sensor_set in sensor_sets:
        # Find vertices
        v0, v1, v2 = _find_vertices(x_sensor[:, np.asarray(sensor_set, dtype=np.int64)], zeta[np.asarray(sensor_set)])

        # Find Centroid by averaging the three vertices
        this_cntr = (v0 + v1 + v2) / 3

        # Accumulate the estimates, we'll divide by num_sets later to make it an average.
        x_est += this_cntr

    return x_est/num_sets


def _parse_sensor_triplets(x_sensor: npt.NDArray[np.float64])-> tuple[list, int]:

    # Parse Inputs
    shp = np.shape(x_sensor)
    num_dim = shp[0] if len(shp) > 0 else 1
    num_sensors = shp[1] if len(shp) > 1 else 1

    if num_dim != 2:
        raise TypeError("Angle Bisector Solution only works in x/y space.")

    # Set up all possible permutations of 3 elements
    if num_sensors <= 15:
        # Get all possible sets of 3 sensors
        sensor_sets = combinations(np.arange(num_sensors), 3)
        num_sets = math.factorial(num_sensors) / (math.factorial(3) * math.factorial(num_sensors - 3))
    else:
        # If there are more than ~15 rows, combinations returns an extremely large set.
        # Instead, just make sequential groups of three
        sensor_sets = [(i, i + 1, i + 2) for i in range(num_sensors - 2)]
        num_sets = len(sensor_sets)

    return sensor_sets, num_sets


def _find_vertices(x: npt.NDArray[np.float64], zeta: npt.NDArray[np.float64])->\
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    # Find vertices
    v0 = find_intersect(x[:, 0], zeta[0], x[:, 1], zeta[1])
    v1 = find_intersect(x[:, 1], zeta[1], x[:, 2], zeta[2])
    v2 = find_intersect(x[:, 2], zeta[2], x[:, 0], zeta[0])

    return v0, v1, v2
