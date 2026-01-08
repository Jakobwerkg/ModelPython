# -*- coding: utf-8 -*-
import numpy as np

from nmwc_model.namelist import (
    idbg,
    idthdt,
    nx,
    nxb,
    nb,
    nz,
    dth,
    dt,
)  # global variables


def prog_isendens(sold, snow, unow, dtdx, dthetadt=None):
    """ Prognostic step for the isentropic mass density.

    Parameters
    ----------
    sold : np.ndarray
        Isentropic density in [kg m^-2 K^-1] defined at the previous time level.
    snow : np.ndarray
        Isentropic density in [kg m^-2 K^-1] defined at the current time level.
    unow : np.ndarray
        Horizontal velocity in [m s^-1] defined at the current time level.
    dtdx : float
        Ratio between timestep in [s] and grid spacing in [m].
    dthetadt : ``np.ndarray``, optional
        Vertical velocity i.e. time derivative of the potential temperature
        (given by the latent heat of condensation/evaporation) in [K s^-1].

    Returns
    -------
    np.ndarray :
        Isentropic density in [kg m^-2 K^-1] defined at the next time level.
    """
    if idbg == 1:
        print("Prognostic step: Isentropic mass density ...\n")

    # Declare
    snew = np.zeros_like(snow)

    # *** Exercise 2.1/5.2 isentropic mass density ***
    # *** time step for isentropic mass density ***
    # *** edit here ***
    #
    i = nb+np.arange(0, nx)
    snew[i,:] = sold[i,:] - 0.5 * dtdx * (snow[i+1,:] * (unow[i+1,:] + unow[i+2,:]) - snow[i-1,:] * (unow[i-1,:] + unow[i,:]))
    
#    if dthetadt is not None:
    if idthdt == 1:

        k = np.arange(1, nz-1)
    
        ii, kk = np.ix_(i, k)

        # dthetadt_centered = 0.5 * (dthetadt[ii, kk] + dthetadt[ii, kk+1])

        # snew[ii, kk] -= dt * dthetadt_centered * (snow[ii, kk+1] - snow[ii, kk-1]) / (2 * dth)
        snew[ii, kk] = snew[ii, kk] - dt / dth * (0.5 * (dthetadt[ii, kk + 1] + dthetadt[ii, kk + 2]) * snow[ii, kk + 1]- 0.5 * (dthetadt[ii, kk - 1] + dthetadt[ii, kk]) * snow[ii, kk - 1])
        
    #
    
    # *** Exercise 2.1/5.2 isentropic mass density ***

    return snew


def prog_velocity(uold, unow, mtg, dtdx, dthetadt=None):
    """ Prognostic step for the momentum.

    Parameters
    ----------
    uold : np.ndarray
        Horizontal velocity in [m s^-1] defined at the previous time level.
    unow : np.ndarray
        Horizontal velocity in [m s^-1] defined at the current time level.
    mtg : np.ndarray
        Montgomery potential in [m^2 s^-2] defined at the current time level.
    dtdx : float
        Ratio between timestep in [s] and grid spacing in [m].
    dthetadt : ``np.ndarray``, optional
        Vertical velocity i.e. time derivative of the potential temperature
        (given by the latent heat of condensation/evaporation) in [K s^-1].

    Returns
    -------
    np.ndarray :
        Horizontal velocity in [m s^-1] defined at the next time level.
    """
    if idbg == 1:
        print("Prognostic step: Velocity ...\n")

    # Declare
    unew = np.zeros_like(unow)

    # *** Exercise 2.1/5.2 velocity ***
    # *** time step for momentum ***
    # *** edit here ***
    #
    i = nb + np.arange(0, nx + 1)
    unew[i,:] = uold[i,:] - unow[i,:] * dtdx * (unow[i+1,:] - unow[i-1,:]) - 2 * dtdx * (mtg[i,:] - mtg[i-1,:])
    
#    if dthetadt is not None:
    if idthdt == 1:
    
        k = np.arange(1, nz-1)

        ii, kk = np.ix_(i,k)

        dthetadt_centered = 0.5 * (dthetadt[ii, kk] + dthetadt[ii, kk+1])

        unew[ii, kk] -= dt * dthetadt_centered * (unow[ii, kk+1] - unow[ii, kk-1]) / (2 * dth)

        #dthetadt_interpolated = 0.25 * (dthetadt[ii, kk] + dthetadt[ii, kk+1] + dthetadt[ii-1, kk] + dthetadt[ii-1, kk+1])
        #unew[ii, kk] = unew[ii, kk] - dt/dth * dthetadt_interpolated * (unow[ii, kk+1] - unow[ii, kk-1])
    #
    # *** Exercise 2.1/5.2 velocity ***

    return unew


def prog_moisture(unow, qvold, qcold, qrold, qvnow, qcnow, qrnow, dtdx, dthetadt=None):
    """ Prognostic step for the hydrometeors.

    Parameters
    ----------
    unow : np.ndarray
        Horizontal velocity in [m s^-1] defined at the current time level.
    qvold : np.ndarray
        Mass fraction of water vapor in [g g^-1] defined at the previous time level.
    qcold : np.ndarray
        Mass fraction of cloud liquid water in [g g^-1] defined at the previous time level.
    qrold : np.ndarray
        Mass fraction of precipitation water in [g g^-1] defined at the previous time level.
    qvnow : np.ndarray
        Mass fraction of water vapor defined in [g g^-1] at the current time level.
    qcnow : np.ndarray
        Mass fraction of cloud liquid water in [g g^-1] defined at the current time level.
    qrnow : np.ndarray
        Mass fraction of precipitation water in [g g^-1] defined at the current time level.
    dtdx : float
        Ratio between timestep in [s] and grid spacing in [m].
    dthetadt : ``np.ndarray``, optional
        Vertical velocity i.e. time derivative of the potential temperature
        (given by the latent heat of condensation/evaporation) in [K s^-1].

    Returns
    -------
    qvnew : np.ndarray
        Mass fraction of water vapor in [g g^-1] defined at the next time level.
    qcnew : np.ndarray
        Mass fraction of cloud liquid water in [g g^-1] defined at the next time level.
    qrnew : np.ndarray
        Mass fraction of precipitation water in [g g^-1] defined at the next time level.
    """

    if idbg == 1:
        print("Prognostic step: Moisture scalars ...\n")

    # Declare
    qvnew = np.zeros_like(qvnow)
    qcnew = np.zeros_like(qcnow)
    qrnew = np.zeros_like(qrnow)

    # *** Exercise 4.1/5.2 moisture advection ***
    # *** edit here ***
    #

    i = nb + np.arange(0, nx)
    u_center = 0.5 * (unow[i, :] + unow[i+1, :])
        
    qvnew[i, :] = qvold[i, :] - dtdx * u_center * (qvnow[i+1, :] - qvnow[i-1, :])
    qcnew[i, :] = qcold[i, :] - dtdx * u_center * (qcnow[i+1, :] - qcnow[i-1, :])
    qrnew[i, :] = qrold[i, :] - dtdx * u_center * (qrnow[i+1, :] - qrnow[i-1, :])
    
#    if dthetadt is not None:
    if idthdt == 1:

        k = np.arange(1, nz-1)

        ii, kk = np.ix_(i, k)

        dthetadt_centered = 0.5 * (dthetadt[ii, kk] + dthetadt[ii, kk+1])

        qvnew[ii, kk] -= dt * dthetadt_centered * (qvnow[ii, kk+1] - qvnow[ii, kk-1]) / (dth)
        qcnew[ii, kk] -= dt * dthetadt_centered * (qcnow[ii, kk+1] - qcnow[ii, kk-1]) / (dth)
        qrnew[ii, kk] -= dt * dthetadt_centered * (qrnow[ii, kk+1] - qrnow[ii, kk-1]) / (dth)
        
#        qvnew[ii, kk] -= dt * dthetadt_centered * (qvnow[ii, kk+1] - qvnow[ii, kk-1]) / (2 * dth)
#        qcnew[ii, kk] -= dt * dthetadt_centered * (qcnow[ii, kk+1] - qcnow[ii, kk-1]) / (2 * dth)
#        qrnew[ii, kk] -= dt * dthetadt_centered * (qrnow[ii, kk+1] - qrnow[ii, kk-1]) / (2 * dth)
        
        
        #qvnew[ii, kk] = qvnew[ii, kk] - dt * 0.5 * (dthetadt[ii, kk] + dthetadt[ii, kk+1]) * (qvnow[ii, kk+1] - qvnow[ii, kk-1]) / dth
        #qcnew[ii, kk] = qcnew[ii, kk] - dt * 0.5 * (dthetadt[ii, kk] + dthetadt[ii, kk+1]) * (qcnow[ii, kk+1] - qcnow[ii, kk-1]) / dth
        #qrnew[ii, kk] = qrnew[ii, kk] - dt * 0.5 * (dthetadt[ii, kk] + dthetadt[ii, kk+1]) * (qrnow[ii, kk+1] - qrnow[ii, kk-1]) / dth
    # *** Exercise 4.1/5.2  ***

    return qvnew, qcnew, qrnew


def prog_numdens(unow, ncold, nrold, ncnow, nrnow, dtdx, dthetadt=None):
    """ Prognostic step for the number densities.

    Parameters
    ----------
    unow : np.ndarray
        Horizontal velocity in [m s^-1] defined at the current time level.
    ncold : np.ndarray
        Number density of cloud liquid water in [g^-1] defined at the previous time level.
    nrold : np.ndarray
        Number density of precipitation water in [g^-1] defined at the previous time level.
    ncnow : np.ndarray
        Number density of cloud liquid water in [g^-1] defined at the current time level.
    nrnow : np.ndarray
        Number density of precipitation water in [g^-1] defined at the current time level.
    dtdx : float
        Ratio between timestep in [s] and grid spacing in [m].
    dthetadt : ``np.ndarray``, optional
        Vertical velocity i.e. time derivative of the potential temperature
        (given by the latent heat of condensation/evaporation) in [K s^-1].

    Returns
    -------
    ncnew : np.ndarray
        Number density of cloud liquid water in [g^-1] defined at the next time level.
    nrnew : np.ndarray
        Number density of precipitation water in [g^-1] defined at the next time level.
    """

    print("ENTER prog_numdens")
    print("  unow max/min:", unow.max(), unow.min())
    print("  ncnow max/min:", ncnow.max(), ncnow.min())
    print("  nrold max/min:", nrold.max(), nrold.min())

    if idbg == 1:
        print("Prognostic step: Number densities ...")

    # Declare
    ncnew = np.zeros_like(ncnow)
    nrnew = np.zeros_like(nrnow)

    # *** Exercise 5.1/5.2 number densities ***
    # *** edit here ***
    #

    i = nb + np.arange(0, nx)
    u_center = 0.5 * (unow[i, :] + unow[i+1, :])

    print("  u_center max/min:", u_center.max(), u_center.min())
    
    ncnew[i, :] = ncold[i, :] - dtdx * u_center * (ncnow[i+1, :] - ncnow[i-1, :])
    nrnew[i, :] = nrold[i, :] - dtdx * u_center * (nrnow[i+1, :] - nrnow[i-1, :])
    
#    if dthetadt is not None:
    if idthdt == 1:

        k = np.arange(1, nz-1) 

        ii, kk = np.ix_(i, k)

        dthetadt_centered = 0.5 * (dthetadt[ii, kk] + dthetadt[ii, kk+1])

        ncnew[ii, kk] -= dt * dthetadt_centered * (ncnow[ii, kk+1] - ncnow[ii, kk-1]) / (dth)
        nrnew[ii, kk] -= dt * dthetadt_centered * (nrnow[ii, kk+1] - nrnow[ii, kk-1]) / (dth)
        
#         ncnew[ii, kk] -= dt * dthetadt_centered * (ncnow[ii, kk+1] - ncnow[ii, kk-1]) / (2 * dth)
#        nrnew[ii, kk] -= dt * dthetadt_centered * (nrnow[ii, kk+1] - nrnow[ii, kk-1]) / (2 * dth)
        
       
        
        #ncnew[ii, kk] = ncnew[ii, kk] - dt * 0.5 * (dthetadt[ii, kk] + dthetadt[ii, kk+1]) * (ncnow[ii, kk+1] - ncnow[ii, kk-1]) / dt
        #nrnew[ii, kk] = nrnew[ii, kk] - dt * 0.5 * (dthetadt[ii, kk] + dthetadt[ii, kk+1]) * (nrnow[ii, kk+1] - nrnow[ii, kk-1]) / dt

#    print("EXIT prog_numdens ncnew max/min:", ncnew.max(), ncnew.min())
#    print("EXIT prog_numdens nrnew max/min:", nrnew.max(), nrnew.min())

    #
    # *** Exercise 5.1/5.2  *

    return ncnew, nrnew
