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

    i = nb + np.arange(0, nx)

    snew[i, :] = sold[i, :]-dtdx*(
        snow[i+1, :]*0.5*(
            unow[i+1, :]+unow[i+2, :]
        ) -
        snow[i-1, :]*0.5*(
            unow[i-1, :]+unow[i, :]
        )
    )

    # *** Exercise 2.1/5.2 isentropic mass density ***
        # --- Vertical advection (Ex. 5.2) ---
    if dthetadt is not None and idthdt == 1:
        k = np.arange(1, nz-1)
        ii, kk = np.ix_(i, k)

        # interpolate dthetadt from interfaces (k, k+1) to full level k
        dthdt_s = 0.5 * (dthetadt[ii, kk] + dthetadt[ii, kk + 1])

        # centered vertical derivative of s
        ds_dth = (snow[ii, kk + 1] - snow[ii, kk - 1]) / (2.0 * dth)

        # add vertical advection tendency
        snew[ii, kk] = snew[ii, kk] - dt * dthdt_s * ds_dth

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

    i = nb + np.arange(0, nx+1)

    unew[i, :] = uold[i, :] - unow[i, :]*dtdx*(
        unow[i+1, :]-unow[i-1, :]
    ) - (
        2*dtdx*(mtg[i, :]-mtg[i-1, :])
    )

        # --- Vertical advection (Ex. 5.2) ---
    if dthetadt is not None and idthdt == 1:
        k = np.arange(1, nz-1)
        ii, kk = np.ix_(i, k)

        # First: interface -> full level at scalar columns
        dthdt_c_im1 = 0.5 * (dthetadt[ii - 1, kk] + dthetadt[ii - 1, kk + 1])  # column i-1
        dthdt_c_i   = 0.5 * (dthetadt[ii,     kk] + dthetadt[ii,     kk + 1])  # column i

        # Second: horizontal average to u-point
        dthdt_u = 0.5 * (dthdt_c_im1 + dthdt_c_i)

        # centered vertical derivative of u
        du_dth = (unow[ii, kk + 1] - unow[ii, kk - 1]) / (2.0 * dth)

        # update
        unew[ii, kk] = unew[ii, kk] - dt * dthdt_u * du_dth

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
    qvnew = np.zeros_like(qvnow) # water vapour
    qcnew = np.zeros_like(qcnow) # cloud liquid
    qrnew = np.zeros_like(qrnow) # rain water

    # *** Exercise 4.1/5.2 moisture advection ***
        
    i = nb + np.arange(0, nx)

    qvnew[i, :] = qvold[i, :] - 0.5 * dtdx * unow[i, :] * (qvnow[i + 1, :] - qvnow[i - 1, :])
    qcnew[i, :] = qcold[i, :] - 0.5 * dtdx * unow[i, :] * (qcnow[i + 1, :] - qcnow[i - 1, :])
    qrnew[i, :] = qrold[i, :] - 0.5 * dtdx * unow[i, :] * (qrnow[i + 1, :] - qrnow[i - 1, :])

    #
    if dthetadt is not None and idthdt == 1:
        k = np.arange(1, nz-1)
        ii, kk = np.ix_(i, k)

        dthdt_s = 0.5 * (dthetadt[ii, kk] + dthetadt[ii, kk + 1])

        dqv_dth = (qvnow[ii, kk + 1] - qvnow[ii, kk - 1]) / (2.0 * dth)
        dqc_dth = (qcnow[ii, kk + 1] - qcnow[ii, kk - 1]) / (2.0 * dth)
        dqr_dth = (qrnow[ii, kk + 1] - qrnow[ii, kk - 1]) / (2.0 * dth)

        qvnew[ii, kk] = qvnew[ii, kk] - dt * dthdt_s * dqv_dth
        qcnew[ii, kk] = qcnew[ii, kk] - dt * dthdt_s * dqc_dth
        qrnew[ii, kk] = qrnew[ii, kk] - dt * dthdt_s * dqr_dth
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

    if idbg == 1:
        print("Prognostic step: Number densities ...")

    # Declare
    ncnew = np.zeros_like(ncnow)
    nrnew = np.zeros_like(nrnow)

    # *** Exercise 5.1/5.2 number densities ***
    i = nb + np.arange(0, nx)

    ncnew[i, :] = ncold[i, :] - 0.5 * dtdx * unow[i, :] * (ncnow[i + 1, :] - ncnow[i - 1, :])
    nrnew[i, :] = nrold[i, :] - 0.5 * dtdx * unow[i, :] * (nrnow[i + 1, :] - nrnow[i - 1, :])

        ## --- Vertical advection (Ex. 5.2) ---
    if dthetadt is not None and idthdt == 1:
        k = np.arange(1, nz-1)
        ii, kk = np.ix_(i, k)

        dthdt_s = 0.5 * (dthetadt[ii, kk] + dthetadt[ii, kk + 1])

        dnc_dth = (ncnow[ii, kk + 1] - ncnow[ii, kk - 1]) / (2.0 * dth)
        dnr_dth = (nrnow[ii, kk + 1] - nrnow[ii, kk - 1]) / (2.0 * dth)

        ncnew[ii, kk] = ncnew[ii, kk] - dt * dthdt_s * dnc_dth
        nrnew[ii, kk] = nrnew[ii, kk] - dt * dthdt_s * dnr_dth
    #
    # *** Exercise 5.1/5.2  *

    return ncnew, nrnew
