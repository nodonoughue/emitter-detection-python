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


# TODO: Unit tests for all converters
# TODO: Unit test for correct_enu
# TODO: Unit test for reckon_sphere_enu

def equal_to_tolerance(x, y, tol=1e-6)->bool:
    """
    Shorthand to compare two lists and ensure each entry has an error less than the specified tolerance
    """
    if len(x) != len(y): return False
    return all([abs(xx-yy)<tol for xx, yy in zip(x,y)])