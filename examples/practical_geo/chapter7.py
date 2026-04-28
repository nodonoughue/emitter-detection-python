import matplotlib.pyplot as plt
import numpy as np

from ewgeo.tdoa import TDOAPassiveSurveillanceSystem
from ewgeo.triang import DirectionFinder
from ewgeo import tracker
from ewgeo.utils import SearchSpace
from ewgeo.utils.constants import speed_of_light
from ewgeo.utils.covariance import CovarianceMatrix
from ewgeo.utils.errors import compute_cep50, draw_error_ellipse
from ewgeo.utils.unit_conversions import convert

_rad2deg = convert(1, "rad", "deg")
_deg2rad = convert(1, "deg", "rad")


def run_all_examples():
    """
    Run all chapter 7 examples and return a list of figure handles

    :return figs: list of figure handles
    """

    return list(example1()) + list(example2()) + list(example3()) + list(example4())


def example1(colors=None):
    """
    Executes Example 7.1.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    28 June 2025

    :return: figure handle to generated graphic
    """

    if colors is None:
        colors = plt.get_cmap("viridis_r")

    # Define sensor positions
    x_aoa = np.array([[0, 1e3],[0, 0]])
    x_tgt = np.array([2e3, 4e3])

    # Define sensor accuracy
    _, n_sensors = np.shape(x_aoa)
    sigma_theta = 5
    sigma_psi = sigma_theta * _deg2rad
    cov_psi = CovarianceMatrix(sigma_psi**2 * np.eye(n_sensors))

    # Make PSS Object
    aoa = DirectionFinder(x=x_aoa, cov=cov_psi, do_2d_aoa=False)

    # Compute CRLB
    crlb = aoa.compute_crlb(x_source=x_tgt)
    cep50= compute_cep50(crlb)
    print('CEP50: {:.2f} km'.format(cep50/1e3))

    cep50_desired = 100
    est_k_min = np.ceil((cep50/cep50_desired)**2)
    print('Estimate {:d} samples required.'.format(np.astype(est_k_min, int)))

    # Iterate over number of samples
    num_samples = np.kron(10**np.arange(5), np.arange(start=1,stop=10))
    sigma_theta_vec = np.array([5, 10, 30])

    cep_vec = np.zeros(shape=(len(sigma_theta_vec), len(num_samples)))
    for idx_s, this_s in enumerate(sigma_theta_vec):
        for idx_n, this_n in enumerate(num_samples):
            this_sigma_psi = this_s * _deg2rad
            this_cov = CovarianceMatrix(this_sigma_psi**2/this_n*np.eye(n_sensors))

            aoa.cov = this_cov
            this_crlb = aoa.compute_crlb(x_source=x_tgt)
            cep_vec[idx_s, idx_n] = compute_cep50(this_crlb)

    fig=plt.figure()
    for this_sigma, this_cep in zip(sigma_theta_vec, cep_vec):
        plt.loglog(num_samples, this_cep, label='$\\sigma_{{\\theta}}={:d}^\\circ$'.format(this_sigma))

    plt.plot(num_samples, 100*np.ones_like(num_samples), 'k--', label='CEP=100 m')
    plt.plot(est_k_min*np.array([1,1]),[1e1,1e4],'k--',label='K={:d}'.format(np.astype(est_k_min, int)))
    plt.ylim([10, 10e3])
    plt.xlabel('Number of Samples [K]')
    plt.ylabel('$CEP_{50}$ [m]')
    plt.grid(True)
    plt.legend(loc='upper left')

    # Determine when CEP50 crosses below 100 m
    desired_cep = 100
    good_samples = cep_vec[0] <= desired_cep  # just compare the 5 deg error case
    good_index = next((i for i, val in enumerate(good_samples) if val), -1)  # search for the first True value; return -1 if none
    if good_index < 0:
        print('More than {:d} samples required to achieve {:.2f} m CEP50.'.format(np.amax(num_samples), desired_cep))
    else:
        print('{:d} samples required to achieve {:.2f} m CEP50.'.format(num_samples[good_index], desired_cep))

    # Plot for AOR
    x_ctr = np.array([0e3, 3e3])
    offset = np.array([5e3, 2e3])
    num_pts = 101
    search_space = SearchSpace(x_ctr=x_ctr,
                               max_offset=offset,
                               points_per_dim=num_pts)
    x_set, x_grid = search_space.x_set, search_space.x_grid
    extent = search_space.get_extent(multiplier=1/1e3)
    # extent = ((x_ctr[0].item() - offset[0])/1e3,
    #           (x_ctr[0].item() + offset[0])/1e3,
    #           (x_ctr[1].item() - offset[1])/1e3,
    #           (x_ctr[1].item() + offset[1])/1e3)

    # Use a squeeze operation to ensure that the individual dimension indices in x_grid are 2D
    x_grid = [np.squeeze(this_dim) for this_dim in x_grid]

    k_vec = [10, 100]
    cmap_lim = [0, 5]

    figs = [fig]
    for this_k in k_vec:
        aoa.cov = cov_psi.multiply(1/this_k, overwrite=False)
        this_crlb = aoa.compute_crlb(x_source=x_set)
        cep = np.reshape(compute_cep50(this_crlb), shape=search_space.grid_shape)

        # Start with the image plot
        this_fig = plt.figure()
        figs.append(this_fig)
        hdl = plt.imshow(cep/1e3, origin='lower', cmap=colors, extent=extent, vmin=cmap_lim[0], vmax=cmap_lim[1])
        plt.colorbar(hdl, format='%d')

        # Add contour lines
        hdl2 = plt.contour(x_grid[0]/1e3, x_grid[1]/1e3, cep/1e3, origin='lower', colors='k')
        plt.clabel(hdl2, fontsize=10, colors='k')
        plt.xlabel('x [km]')
        plt.ylabel('y [km]')

        aoa.plot_sensors(scale=1e3,marker='^',label='Sensors')
        plt.ylim([-.5, 5.5])

    return figs


def example2():
    """
    Executes Example 7.2.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    28 June 2025

    :return: figure handle to generated graphic
    """

    # Define sensor positions
    x_tdoa = np.array([[-10e3, 0, 10e3],
                       [0, 10e3, 0]])
    x_tgt = np.array([5e3, -15e3])
    
    # Define sensor accuracy
    _, n_sensors = np.shape(x_tdoa)
    sigma_t = 1e-6
    cov_toa = CovarianceMatrix(sigma_t**2*np.eye(n_sensors))
    cov_roa = cov_toa.multiply(speed_of_light ** 2, overwrite=False)
    ref_idx = None  # use default reference sensor

    # Initialize PSS Object
    tdoa = TDOAPassiveSurveillanceSystem(x=x_tdoa, cov=cov_roa, variance_is_toa=False, ref_idx=ref_idx)

    # Define pulse timing
    pri = 1e-3
    T = 1
    num_pulses = np.floor(T/pri)+1
    
    # Compute CRLB
    crlb_single_sample = tdoa.compute_crlb(x_source=x_tgt)
    crlb_sample_mean = tdoa.compute_crlb(x_source=x_tgt,
                                         cov=tdoa.cov.multiply(1/num_pulses, overwrite=False))
    cep_single_sample = compute_cep50(crlb_single_sample)
    cep_sample_mean = compute_cep50(crlb_sample_mean)
    print('CEP50 for a single sample: {:.2f} km'.format(cep_single_sample/1e3))
    print('CEP50 for sample mean (K=1001): {:.2f} m'.format(cep_sample_mean))

    num_pulses_needed = np.ceil((cep_single_sample / 10) ** 2).astype(int)
    print(f"Predict {num_pulses_needed} samples needed to achieve 10m CEP.")

    # Iterate over observation interval
    sample_vec = np.arange(50e3,step=10)+1
    sigma_t_vec = np.array([.1e-6, 1e-6, 10e-6])
    cep_vec = np.zeros((len(sigma_t_vec), len(sample_vec)))
    for idx_s, this_sigma_t in enumerate(sigma_t_vec):
        this_sigma_t = np.array(this_sigma_t)
        this_cov_roa = (this_sigma_t**2 * speed_of_light **2) * np.eye(n_sensors)
        tdoa.cov = this_cov_roa

        crlb_vec = [tdoa.compute_crlb(x_source=x_tgt,
                                      cov=tdoa.cov.multiply(1/k, overwrite=False)) for k in sample_vec]
        cep_vec[idx_s] = np.asarray([compute_cep50(c) for c in crlb_vec])

    fig1=plt.figure()
    time_vec = sample_vec * pri
    for this_cep, this_sigma_t in zip(cep_vec, sigma_t_vec):
        plt.loglog(time_vec, this_cep, label='$\\sigma_t={:.1f} \\mu s$'.format(this_sigma_t*1e6))

    plt.plot(time_vec, 10*np.ones_like(time_vec),'k-.', label='10 m')
    plt.xlabel('Time [s]')
    plt.ylabel('$CEP_{50}$ [m]')
    plt.ylim([1, 1000])
    plt.legend(loc='upper right')
    plt.grid(True)

    # Determine when CEP50 crosses below 10 m
    desired_cep = 10
    good_samples = cep_vec[1] <= desired_cep  # just compare the 1 microsecond error case
    good_index = next((i for i, val in enumerate(good_samples) if val), -1)  # search for the first True value; return -1 if none
    if good_index < 0:
        print('More than {:.2f} s required to achieve {:.2f} m CEP50.'.format(np.amax(time_vec), desired_cep))
    else:
        print('{:.2f} s (K={:d}) required to achieve {:.2f} m CEP50.'.format(time_vec[good_index], good_index+1, desired_cep))

    int_time = time_vec[good_index]
    num_pulses_good = np.astype(np.floor(int_time/pri)+1, int).item()
    
    # Reset the covariance matrix
    tdoa.cov = (sigma_t**2 * speed_of_light **2) * np.eye(n_sensors) # reset the covariance matrix
    num_pulses = np.astype(np.floor(1/pri)+1, int).item() # 1 second observation time

    # Demonstrate geolocation
    zeta = tdoa.noisy_measurement(x_source=x_tgt, num_samples=num_pulses)

    # Sample Mean
    zeta_mn = np.cumsum(zeta,axis=1)/(1+np.arange(num_pulses))
    
    # Geolocation Result
    x_ls = np.zeros(shape=(len(x_tgt), num_pulses))
    x_ls_mn = np.zeros(shape=(len(x_tgt), num_pulses))
    
    x_init = np.array([1, 1])*1e3
    
    for idx in np.arange(num_pulses):
        this_x_ls, _ = tdoa.least_square(zeta=zeta[:, idx], x_init=x_init)
        this_x_ls_mn, _ = tdoa.least_square(zeta=zeta_mn[:, idx], x_init=x_init,
                                             cov=tdoa.cov.multiply(1/(idx+1), overwrite=False))
        x_ls[:, idx] = this_x_ls
        x_ls_mn[:, idx] = this_x_ls_mn
    
    fig2 = plt.figure()
    # plt.plot(x_tdoa[0]/1e3, x_tdoa[1]/1e3,'o',label='TDOA Sensors')
    plt.plot(x_tgt[0]/1e3, x_tgt[1]/1e3, '^', label='Target')
    # plt.plot(x_ls[0]/1e3,x_ls[1]/1e3, '--', label='LS Soln (single sample)')
    plt.plot(x_ls_mn[0,:num_pulses_good]/1e3,x_ls_mn[1,:num_pulses_good]/1e3, '-.', label='LS Soln (sample mean)')
    plt.grid(True)

    # Overlay error ellipse
    ell = draw_error_ellipse(x_tgt, crlb_single_sample, num_pts=101)
    ell_full = draw_error_ellipse(x_tgt, crlb_sample_mean, num_pts=101)
    
    plt.plot(ell[0]/1e3, ell[1]/1e3, label='Error Ellipse (single sample)')
    plt.plot(ell_full[0]/1e3, ell_full[1]/1e3, label='Error Ellipse (sample mean)')
    plt.legend(loc='upper right')
    plt.xlabel('x [km]')
    plt.ylabel('y [km]')

    # Plot error as a function of time
    err = np.sqrt(np.sum(np.fabs(x_ls-x_tgt[:, np.newaxis])**2, axis=0))
    err_mn = np.sqrt(np.sum(np.fabs(x_ls_mn-x_tgt[:, np.newaxis])**2, axis=0))

    fig3 = plt.figure()
    time_vec = pri * (1+np.arange(num_pulses))
    plt.semilogy(time_vec, err, label='Error (single sample)')
    plt.semilogy(time_vec, err_mn, label='Error (sample mean)')
    plt.semilogy(time_vec, cep_single_sample*np.ones_like(time_vec), label='CRLB (single sample)')
    plt.semilogy(time_vec, cep_single_sample/np.sqrt(1+np.arange(num_pulses)), label='CRLB (sample mean)')
    plt.xlabel('Time [s]')
    plt.ylabel('Error [m]')
    plt.ylim([10, 10e3])
    plt.legend(loc='upper right')
    plt.grid(True)
    
    return fig1, fig2, fig3


def example3():
    """
    Executes Example 7.3.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    28 June 2025

    :return: figure handle to generated graphic
    """
    # Define sensor positions
    x_aoa = np.array([[-5e3, 5e3], [0, 0]])
    x_tgt = np.array([ 3e3, 25e3])

    # Define sensor accuracy
    _, n_sensors = np.shape(x_aoa)
    sigma_theta = 10
    sigma_psi = sigma_theta*_deg2rad
    cov_psi = CovarianceMatrix(sigma_psi**2*np.eye(n_sensors))

    aoa = DirectionFinder(x=x_aoa, cov=cov_psi, do_2d_aoa=False)

    # Define pulse timing
    pri = 1e-3
    int_time = 30 # observation period
    num_pulses = np.floor(int_time/pri).astype(int)+1

    # Generate noisy measurements
    zeta = aoa.noisy_measurement(x_source=x_tgt, num_samples=num_pulses)

    # Define measurement and Jacobian functions via MeasurementModel
    _ss_pos = tracker.CartesianStateSpace(num_dims=aoa.num_dim, has_vel=False)
    _mm = tracker.MeasurementModel(pss=aoa)

    # Estimate position recursively, using EKF Update algorithm
    x_est = np.zeros(shape=(aoa.num_dim, num_pulses))
    cep = np.zeros(shape=(num_pulses,))

    x_init = np.array([1, 1]) * 1e3
    for idx in np.arange(num_pulses):
        # Grab the current measurement
        this_zeta = zeta[:, idx]

        if idx==0:
            # Initialization
            this_x, _ = aoa.least_square(zeta=this_zeta, x_init=x_init)
            this_p = aoa.compute_crlb(x_source=this_x)
            this_state = tracker.State(state_space=_ss_pos, time=0, state=this_x, covar=this_p)
        else:
            # EKF Update
            this_state = tracker.ekf_update(prev_state, this_zeta, aoa.cov, _mm.measurement, _mm.jacobian)

        # Store the results and update the variables
        x_est[:, idx] = this_state.position
        cep[idx] = compute_cep50(this_state.covar)

        prev_state = this_state

    fig1=plt.figure()
    plt.scatter(x_tgt[0]/1e3, x_tgt[1]/1e3, marker='^', label='Target')
    plt.plot(x_est[0]/1e3, x_est[1]/1e3, '-.', label='Estimated Position')
    plt.grid(True)

    # Draw Error Ellipse from single sample
    crlb = aoa.compute_crlb(x_tgt)
    ell = draw_error_ellipse(x_tgt, crlb, num_pts=101)
    plt.plot(ell[0]/1e3, ell[1]/1e3,'-.', label='Error Ellipse (single msmt.)')

    crlb_adjust = crlb.multiply(1/np.sqrt(num_pulses), overwrite=False)
    ell_1s = draw_error_ellipse(x_tgt, crlb_adjust, num_pts=101)
    plt.plot(ell_1s[0]/1e3, ell_1s[1]/1e3,'-.', label='Error Ellipse (full observation)')

    offset = np.amax(np.amax(ell, axis=1)-np.amin(ell,axis=1), axis=0)
    plt.xlim(x_tgt[0]/1e3 + .6/1e3*offset*np.array([-1, 1]))
    plt.ylim(x_tgt[1]/1e3 + .6/1e3*offset*np.array([-1, 1]))
    plt.legend(loc='upper right')
    plt.xlabel('x [km]')
    plt.ylabel('y [km]')

    # Compute Errors
    err = np.sqrt(np.sum(np.fabs(x_est - x_tgt[:, np.newaxis])**2, axis=0))
    time_vec = pri*(1+np.arange(num_pulses))

    fig2=plt.figure()
    plt.semilogy(time_vec, err, label='Measured')
    plt.plot(time_vec, cep, label='$Predicted (CEP_{50})$')
    plt.plot(time_vec, 100*np.ones_like(time_vec), 'k-.', label='100 m')
    plt.xlabel('Time [s]')
    plt.ylabel('Error [m]')
    plt.legend(loc='upper right')

    figs = [fig1, fig2]

    # Find Time to Achieved Desired Error
    desired_cep = 100
    good_samples = cep <= desired_cep  # just compare the 5 deg error case
    good_index = next((i for i, val in enumerate(good_samples) if val), -1)  # search for the first True value; return -1 if none
    if good_index < 0:
        print('More than {:.2f} s required to achieve {:.2f} m CEP50.'.format(time_vec[-1], desired_cep))
    else:
        print('{:.2f} s required to achieve {:.2f} m CEP50.'.format(time_vec[good_index], desired_cep))

    return figs


def example4():
    """
    Executes Example 7.4.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    28 June 2025

    :return: figure handle to generated graphic
    """
    # Define sensor positions
    x_aoa = np.array([[-5e3, 5e3],[0, 0]])
    v_aoa = np.array([[0, 0],[200, 200]])
    x_tgt = np.array([3e3, 25e3])
    
    # Define sensor accuracy
    num_dims, num_sensors = np.shape(x_aoa)
    sigma_theta = 10
    sigma_psi = sigma_theta*_deg2rad
    cov_psi = CovarianceMatrix((sigma_psi**2)*np.eye(num_sensors))

    # Make PSS Object
    aoa = DirectionFinder(x=x_aoa, cov=cov_psi, do_2d_aoa=False)

    # Define pulse timing
    pri = 1e-3
    int_time = 30 # observation period
    num_pulses = np.floor(int_time/pri).astype(int)+1
    
    # Define outputs
    x_est = np.zeros((aoa.num_dim, num_pulses))
    cep = np.zeros((num_pulses, ))
    x_init = np.array([1, 1])*1e3
    
    # Step through pulses
    this_p=None
    _ss_pos = tracker.CartesianStateSpace(num_dims=aoa.num_dim, has_vel=False)
    _mm = tracker.MeasurementModel(pss=aoa)

    for idx in np.arange(num_pulses):
        # Update positions
        this_x_aoa = x_aoa + v_aoa * idx * pri
        aoa.pos = this_x_aoa

        # Generate noisy measurements
        zeta = aoa.noisy_measurement(x_source=x_tgt)

        if idx==0:
            # Initialization
            this_x, _ = aoa.least_square(zeta=zeta, x_init=x_init)
            this_p = aoa.compute_crlb(x_source=this_x)
            this_state = tracker.State(state_space=_ss_pos, time=0., state=this_x, covar=this_p)
        else:
            # EKF Update
            this_state = tracker.ekf_update(prev_state, zeta, aoa.cov, _mm.measurement, _mm.jacobian)

        # Store the results and update the variables
        x_est[:, idx] = this_state.position
        cep[idx] = compute_cep50(this_state.position_covar)

        prev_state = this_state

    fig1=plt.figure()
    plt.plot(x_tgt[0]/1e3,x_tgt[1]/1e3,'^', label='Target')
    plt.plot(x_est[0]/1e3,x_est[1]/1e3,'-.', label='Estimated Position')
    plt.grid(True)

    # Draw Error Ellipse from single sample
    crlb = aoa.compute_crlb(x_tgt)
    ell = draw_error_ellipse(x=x_tgt, covariance=crlb, num_pts=101)
    plt.plot(ell[0]/1e3, ell[1]/1e3,'-.',label='Error Ellipse (single msmt.)')
    
    ell_end = draw_error_ellipse(x=x_tgt, covariance=prev_state.covar, num_pts=101)
    plt.plot(ell_end[0]/1e3, ell_end[1]/1e3,'-.',label='Error Ellipse (Final EKF Update)')
    
    offset = np.amax(np.amax(ell, axis=1)-np.amin(ell,axis=1), axis=None)
    plt.xlim(x_tgt[0]/1e3 + .6*offset/1e3*np.array([-1, 1]))
    plt.ylim(x_tgt[1]/1e3 + .6*offset/1e3*np.array([-1, 1]))
    plt.xlabel('x [km]')
    plt.ylabel('y [km]')
    plt.legend(loc='upper right')

    ## Compute Errors
    err = np.sqrt(np.sum(np.fabs(x_est - x_tgt[:, np.newaxis])**2, axis=0))
    time_vec = pri*(1+np.arange(num_pulses))
    
    fig2=plt.figure()
    plt.semilogy(time_vec,err/1e3,label='Measured')
    plt.plot(time_vec,cep/1e3,label='$Predicted (CEP_{50})$')
    plt.xlabel('Time [s]')
    plt.ylabel('Error [km]')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    ## Find Time to Achieved Desired Error
    desired_cep = 100
    good_samples = cep <= desired_cep  
    good_index = next((i for i, val in enumerate(good_samples) if val), -1)  # search for the first True value; return -1 if none
    if good_index < 0:
        print('More than {:.2f} s required to achieve {:.2f} m CEP50.'.format(np.amax(time_vec), desired_cep))
    else:
        print('{:.2f} s required to achieve {:.2f} m CEP50.'.format(time_vec[good_index], desired_cep))    
    
    return fig1, fig2


if __name__ == '__main__':
    run_all_examples()
    plt.show()
