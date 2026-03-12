import ewgeo.utils.coordinates as coordinates


def test_homogeneous_transforms():
    """
    Make sure that if we go one way and then back, we get the original result
    """

    # AER<->ECEF
    aer_init = [45, 10, 1000]
    lla_ref = [20, 10, 0]

    ecef = coordinates.aer_to_ecef(aer_init[0], aer_init[1], aer_init[2],
                                   lla_ref[0], lla_ref[1], lla_ref[2],
                                   angle_units='deg', dist_units='m')

    aer_out = coordinates.ecef_to_aer(ecef[0], ecef[1], ecef[2],
                                      lla_ref[0], lla_ref[1], lla_ref[2],
                                      angle_units='deg', dist_units='m')

    assert equal_to_tolerance(aer_init, aer_out), 'Error testing AER->ECEF->AER transform'

    # AER<->LLA
    aer_init = [45, 10, 1000]
    lla_ref = [20, 10, 0]

    lla = coordinates.aer_to_lla(aer_init[0], aer_init[1], aer_init[2],
                                 lla_ref[0], lla_ref[1], lla_ref[2],
                                 angle_units='deg', dist_units='m')

    aer_out = coordinates.lla_to_aer(lla[0], lla[1], lla[2],
                                     lla_ref[0], lla_ref[1], lla_ref[2],
                                     angle_units='deg', dist_units='m')

    assert equal_to_tolerance(aer_init, aer_out), 'Error testing AER->LLA->AER transform'

    # ECEF<->AER
    ecef_init = [4198945, 174747, 4781887] # Parc des Buttes-Chaumont
    lla_ref = [48.0124, 2.5451, 163.4885] # Sensor at Charles de Gaulle airport

    aer = coordinates.ecef_to_aer(ecef_init[0], ecef_init[1], ecef_init[2],
                                 lla_ref[0], lla_ref[1], lla_ref[2],
                                 angle_units='deg', dist_units='m')

    ecef_out = coordinates.aer_to_ecef(aer[0], aer[1], aer[2],
                                       lla_ref[0], lla_ref[1], lla_ref[2],
                                       angle_units='deg', dist_units='m')

    assert equal_to_tolerance(ecef_init, ecef_out), 'Error testing ECEF->AER->ECEF transform'

    # ECEF<->ENU
    ecef_init = [4198945, 174747, 4781887]  # Parc des Buttes-Chaumont
    lla_ref = [48.0124, 2.5451, 163.4885]  # Sensor at Charles de Gaulle airport

    enu = coordinates.ecef_to_enu(ecef_init[0], ecef_init[1], ecef_init[2],
                                  lla_ref[0], lla_ref[1], lla_ref[2],
                                  angle_units='deg', dist_units='m')

    ecef_out = coordinates.enu_to_ecef(enu[0], enu[1], enu[2],
                                       lla_ref[0], lla_ref[1], lla_ref[2],
                                       angle_units='deg', dist_units='m')

    assert equal_to_tolerance(ecef_init, ecef_out), 'Error testing ECEF->ENU->ECEF transform'

    # ECEF Velocity<->ENU Velocity
    ecef_init_vel = [100., 200., 150.]
    lla_ref = [48.0124, 2.5451, 163.4885]  # Sensor at Charles de Gaulle airport

    enu_vel = coordinates.ecef_to_enu_vel(ecef_init_vel[0], ecef_init_vel[1], ecef_init_vel[2],
                                          lla_ref[0], lla_ref[1],
                                          angle_units='deg')

    ecef_out_vel = coordinates.enu_to_ecef_vel(enu_vel[0], enu_vel[1], enu_vel[2],
                                               lla_ref[0], lla_ref[1],
                                               angle_units='deg')

    assert equal_to_tolerance(ecef_init_vel, ecef_out_vel), 'Error testing ECEF->ENU->ECEF (velocity) transform'

    # ECEF<->LLA
    ecef_init = [4198945, 174747, 4781887]  # Parc des Buttes-Chaumont

    lla = coordinates.ecef_to_lla(ecef_init[0], ecef_init[1], ecef_init[2],
                                  angle_units='deg', dist_units='m')

    ecef_out = coordinates.lla_to_ecef(lla[0], lla[1], lla[2],
                                       angle_units='deg', dist_units='m')

    assert equal_to_tolerance(ecef_init, ecef_out), 'Error testing ECEF->LLA->ECEF transform'

    # ENU<->AER
    enu_init = [150, 75, 200]

    aer = coordinates.enu_to_aer(enu_init[0], enu_init[1], enu_init[2],
                                 angle_units='deg')

    enu_out = coordinates.aer_to_enu(aer[0], aer[1], aer[2],
                                     angle_units='deg')

    assert equal_to_tolerance(enu_init, enu_out), 'Error testing ENU->AER->ENU transform'

    # ENU<->ECEF
    enu_init = [150, 75, 200]
    lla_ref = [48.0124, 2.5451, 163.4885]  # Sensor at Charles de Gaulle airport

    ecef = coordinates.enu_to_ecef(enu_init[0], enu_init[1], enu_init[2],
                                   lla_ref[0], lla_ref[1], lla_ref[2],
                                   angle_units='deg', dist_units='m')

    enu_out = coordinates.ecef_to_enu(ecef[0], ecef[1], ecef[2],
                                      lla_ref[0], lla_ref[1], lla_ref[2],
                                      angle_units='deg', dist_units='m')

    assert equal_to_tolerance(enu_init, enu_out), 'Error testing ENU->ECEF->ENU transform'

    # ENU Velocity<->ECEF Velocity
    enu_init_vel = [100., 200., 150.]
    lla_ref = [48.0124, 2.5451, 163.4885]  # Sensor at Charles de Gaulle airport

    ecef_vel = coordinates.enu_to_ecef_vel(enu_init_vel[0], enu_init_vel[1], enu_init_vel[2],
                                           lla_ref[0], lla_ref[1],
                                           angle_units='deg')

    enu_out_vel = coordinates.ecef_to_enu_vel(ecef_vel[0], ecef_vel[1], ecef_vel[2],
                                              lla_ref[0], lla_ref[1],
                                              angle_units='deg')

    assert equal_to_tolerance(enu_init_vel, enu_out_vel), 'Error testing ENU->ECEF->ENU (velocity) transform'

    # ENU<->LLA
    enu_init = [150, 75, 200]
    lla_ref = [48.0124, 2.5451, 163.4885]  # Sensor at Charles de Gaulle airport

    lla = coordinates.enu_to_lla(enu_init[0], enu_init[1], enu_init[2],
                                 lla_ref[0], lla_ref[1], lla_ref[2],
                                 angle_units='deg', dist_units='m')

    enu_out = coordinates.lla_to_enu(lla[0], lla[1], lla[2],
                                     lla_ref[0], lla_ref[1], lla_ref[2],
                                     angle_units='deg', dist_units='m')

    assert equal_to_tolerance(enu_init, enu_out), 'Error testing ENU->ECEF->ENU transform'

    # LLA<->AER
    lla_init = [48.8800, 2.3831, 124.5089]  # Parc des Buttes-Chaumont
    lla_ref = [48.0124, 2.5451, 163.4885]  # Sensor at Charles de Gaulle airport

    aer = coordinates.lla_to_aer(lla_init[0], lla_init[1], lla_init[2],
                                 lla_ref[0], lla_ref[1], lla_ref[2],
                                 angle_units='deg', dist_units='m')

    lla_out = coordinates.aer_to_lla(aer[0], aer[1], aer[2],
                                     lla_ref[0], lla_ref[1], lla_ref[2],
                                     angle_units='deg', dist_units='m')

    assert equal_to_tolerance(lla_init, lla_out), 'Error testing LLA->AER->LLA transform'

    # LLA<->ECEF
    lla_init = [48.8800, 2.3831, 124.5089]  # Parc des Buttes-Chaumont

    ecef = coordinates.lla_to_ecef(lla_init[0], lla_init[1], lla_init[2],
                                   angle_units='deg', dist_units='m')

    lla_out = coordinates.ecef_to_lla(ecef[0], ecef[1], ecef[2],
                                      angle_units='deg', dist_units='m')

    assert equal_to_tolerance(lla_init, lla_out), 'Error testing LLA->ECEF->LLA transform'

    # LLA<->ENU
    lla_init = [48.8800, 2.3831, 124.5089]  # Parc des Buttes-Chaumont
    lla_ref = [48.0124, 2.5451, 163.4885]  # Sensor at Charles de Gaulle airport

    enu = coordinates.lla_to_enu(lla_init[0], lla_init[1], lla_init[2],
                                 lla_ref[0], lla_ref[1], lla_ref[2],
                                 angle_units='deg', dist_units='m')

    lla_out = coordinates.enu_to_lla(enu[0], enu[1], enu[2],
                                     lla_ref[0], lla_ref[1], lla_ref[2],
                                     angle_units='deg', dist_units='m')

    assert equal_to_tolerance(lla_init, lla_out), 'Error testing LLA->ENU->LLA transform'


def test_global_unit_conversion():
    """
    Test all global coordinate conversions for a few key points
    """
    zero_point = {'lla': [0., 0., 0.],
                  'ecef': [6378137.0, 0., 0.]}
    north_pole = {'lla': [90., 0., 0.],
                  'ecef': [0., 0., 6356752.3]}
    paris = {'lla': [48.8562, 2.3508, 67.4],
             'ecef': [4200996.8, 172460.3, 4780102.8]}
    death_valley = {'lla': [36.4594, -116.8291, -86.0],
                    'ecef': [-2317945.4, -4582966.8, 3769260.9]}

    for coord in [zero_point, north_pole, paris, death_valley]:
        ecef_out = coordinates.lla_to_ecef(coord['lla'][0], coord['lla'][1], coord['lla'][2])
        assert equal_to_tolerance(coord['ecef'], ecef_out, tol=0.1), 'Error testing LLA->ECEF transform'

        lla_out = coordinates.ecef_to_lla(coord['ecef'][0], coord['ecef'][1], coord['ecef'][2])
        assert equal_to_tolerance(coord['lla'], lla_out, tol=0.1), 'Error testing ECEF->LLA transform'


def test_local_unit_conversion():
    """
    Test AER/ENU unit conversions.
    """
    east = {'enu': [100, 0, 0],
            'aer': [90, 0, 100]}
    north = {'enu': [0, 100, 0],
             'aer': [0, 0, 100]}
    up = {'enu': [0, 0, 100],
          'aer': [0, 90, 100]}
    nonzero = {'enu': [100, 100, 100],
               'aer': [45, 35.2644, 173.2051]}

    for coord in [east, north, up, nonzero]:
        aer_out = coordinates.enu_to_aer(coord['enu'][0], coord['enu'][1], coord['enu'][2])
        # print('ENU->AER Output...')
        # print(aer_out)
        # print(coord['aer'])
        assert equal_to_tolerance(coord['aer'], aer_out, tol=1e-2), 'Error testing ENU->AER transform'

        enu_out = coordinates.aer_to_enu(coord['aer'][0], coord['aer'][1], coord['aer'][2])
        # print('AER->ENU Output...')
        # print(enu_out)
        # print(coord['enu'])
        assert equal_to_tolerance(coord['enu'], enu_out, tol=1e-1), 'Error testing AER->ENU transform'

    return


def test_relative_unit_conversion():
    """
    Test ECEF/ENU unit conversions, using local reference points. Each test point for conversion has a
    Lat/Lon/Alt reference point, an ENU offset, and an ECEF coordinate for the destination of that offset.

    """
    origin_1 = {'lla': [0., 0., 0.],
                'enu': [0., 0., 0.],
                'ecef': [6378137.0, 0., 0.]}
    north_pole_1 = {'lla': [90., 0., 0.],
                    'enu': [0., 0., 0.],
                    'ecef': [0., 0., 6356752.3142]}
    paris_1 = {'lla': [48.8562, 2.3508, 67.4],
               'enu': [10., 0., 0.],
               'ecef': [4200996.4, 172470.3, 4780102.8]}
    paris_2 = {'lla': [48.8562, 2.3508, 67.4],
               'enu': [0., 100., 0.],
               'ecef': [4200921.5, 172457.2, 4780168.6]}
    paris_3  = {'lla': [48.8562, 2.3508, 67.4],
               'enu': [0., 0., 1000.],
               'ecef': [4201654.2, 172487.3, 4780855.9]}
    death_valley_1 = {'lla': [36.4594, -116.8291, -86.0],
                      'enu': [0., 0., 0.],
                      'ecef': [-2317945.4, -4582966.8, 3769260.9]}

    for coord in [origin_1, north_pole_1, paris_1, paris_2, paris_3, death_valley_1]:
        enu_out = coordinates.ecef_to_enu(x=coord['ecef'][0], y=coord['ecef'][1], z=coord['ecef'][2],
                                         lat_ref=coord['lla'][0], lon_ref=coord['lla'][1], alt_ref=coord['lla'][2])
        # print('ECEF->ENU Output...')
        # print(enu_out)
        # print(coord['enu'])
        assert equal_to_tolerance(coord['enu'], enu_out, tol=1e-1), 'Error testing ECEF->ENU transform'

        ecef_out = coordinates.enu_to_ecef(east=coord['enu'][0], north=coord['enu'][1], up=coord['enu'][2],
                                          lat_ref=coord['lla'][0], lon_ref=coord['lla'][1], alt_ref=coord['lla'][2])
        # print('ENU->ECEF Output...')
        # print(ecef_out)
        # print(coord['ecef'])
        assert equal_to_tolerance(coord['ecef'], ecef_out, tol=1e-1), 'Error testing ENU->ECEF transform'

# TODO: Unit test for correct_enu
# TODO: Unit test for reckon_sphere_enu

def equal_to_tolerance(x, y, tol=1e-6)->bool:
    """
    Shorthand to compare two lists and ensure each entry has an error less than the specified tolerance
    """
    if len(x) != len(y): return False
    return all([abs(xx-yy)<tol for xx, yy in zip(x,y)])