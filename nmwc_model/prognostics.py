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
    
    i = nb + np.arange(0, nx+1)
    qvnew[i, :] = qvold[i, :] - dtdx * (
    unow[i, :] * (qvnow[i+1, :] - qvnow[i-1, :])
    )
    qcnew[i, :] = qcold[i, :] - dtdx * (
    unow[i, :] * (qcnow[i+1, :] - qcnow[i-1, :])
    )
    qrnew[i, :] = qrold[i, :] - dtdx * (
    unow[i, :] * (qrnow[i+1, :] - qrnow[i-1, :])
    )

    #
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
    i = nb + np.arange(0, nx + 1)

    ncnew[i, :] = ncold[i, :] - dtdx * (
        unow[i, :] * (ncnow[i + 1, :] - ncnow[i - 1, :])
    )

    nrnew[i, :] = nrold[i, :] - dtdx * (
        unow[i, :] * (nrnow[i + 1, :] - nrnow[i - 1, :])
    )

    #

    #
    # *** Exercise 5.1/5.2  *

    return ncnew, nrnew
