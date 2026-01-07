#!/usr/bin/env python
# -*- coding: utf-8 -*-

from argparse import ArgumentParser, RawDescriptionHelpFormatter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.ticker import MultipleLocator

from nmwc_model.readsim import readsim
from nmwc_model.xzplot import plot_dict


def arg_parser():
    usage = "usage: %(prog)s [options] <filename.npz>\n" \
            "Basic: %(prog)s output_ex3.1.2_100h.npz\n" \
            "Example: %(prog)s -o anim.mp4 --vci 2 --fps 20 output_ex3.1.2_100h.npz"

    description = """
    Produces an animation (x,z) of horizontal velocity using the same
    color scaling philosophy as hovx_vel.py (symmetric around u00, vci-based levels).
    """

    op = ArgumentParser(
        usage=usage,
        description=description,
        formatter_class=RawDescriptionHelpFormatter,
    )

    op.add_argument("filename", metavar="filename", nargs=1, type=str,
                    help="NPZ file holding the data from the model")

    op.add_argument("-o", dest="outname", default="xz_vel_animation.mp4",
                    help="Name of the output animation (mp4)", metavar="FILE.mp4")

    op.add_argument("--fps", dest="fps", default=20, type=int,
                    help="Frames per second", metavar="20")

    op.add_argument("--dpi", dest="dpi", default=150, type=int,
                    help="DPI for the saved movie", metavar="150")

    op.add_argument("--vci", dest="vci", default=2, type=int,
                    help="Velocity contouring interval [m/s]", metavar="2")

    op.add_argument(
        "--vlim",
        dest="vlim",
        default=(0.0, 60.0),
        nargs=2,
        metavar=("0", "60"),
        help="restrict the velocity contours",
        type=float,
    )

    op.add_argument("--every", dest="every", default=1, type=int,
                    help="Use every Nth output time for frames (speeds up long runs)", metavar="1")

    return op


def compute_levels(U_all, u00, vci, vlim=None):
    """Mimic hovx_vel.py style: choose integer levels around u00 with spacing vci,
    and use symmetric normalization around u00."""
    if vlim is not None:
        data_min = vlim[0]
        data_max = vlim[1]
    else:
        data_min = np.nanmin(U_all)
        data_max = np.nanmax(U_all)

    vciDiff = int((u00 - data_min) / vci + 0.5)
    vMinInt = u00 - vciDiff * vci

    vciDiff = int((data_max - u00) / vci + 0.5)
    vMaxInt = u00 + vciDiff * vci

    clev = np.arange(vMinInt, vMaxInt + vci, vci)
    ticks = np.arange(clev[0], clev[-1] + vci, vci)
    valRange = np.arange(clev[0] - 0.5 * vci, clev[-1] + 1.5 * vci, vci)

    distUpMid = clev[-1] + 1.5 * vci - u00
    distMidDown = u00 - clev[0] - 0.5 * vci
    maxDist = max(distUpMid, distMidDown)
    vmin = u00 - maxDist
    vmax = u00 + maxDist

    return valRange, ticks, vmin, vmax


def main():
    op = arg_parser()
    args = op.parse_args()

    varname = "horizontal_velocity"
    var = readsim(args.filename[0], varname)
    pd = plot_dict(args, var, varname)

    # Expect shape: (nt, nz, nx)
    U = np.asarray(var.horizontal_velocity)
    if U.ndim != 3:
        raise RuntimeError(f"Expected 3D array (time,z,x) for {varname}, got shape {U.shape}")

    # Subsample frames if requested
    every = max(1, int(args.every))
    frame_ids = np.arange(0, U.shape[0], every)
    U_anim = U[frame_ids, :, :]

    # Axes: x in km from var.x (hovx_vel uses km)
    x_km = np.asarray(var.x)

    # Vertical coordinate: try common attributes; fall back to index
    z_km = None
    for cand in ["z", "zlev", "height", "z_km"]:
        if hasattr(var, cand):
            z_km = np.asarray(getattr(var, cand))
            break
    if z_km is None:
        z_km = np.arange(U.shape[1])  # index

    # If z is staggered or mismatched length, fall back to index
    if z_km.ndim != 1 or len(z_km) != U.shape[1]:
        z_km = np.arange(U.shape[1])

    # Compute fixed levels and symmetric normalization (consistent across frames)
    vlim = tuple(args.vlim) if args.vlim is not None else None
    valRange, ticks, vmin, vmax = compute_levels(U_anim, var.u00, args.vci, vlim=vlim)

    fig, ax = plt.subplots(1)

    # First frame contourf
    cs = ax.contourf(
        x_km,
        z_km,
        U_anim[0, :, :],
        valRange,
        vmin=vmin,
        vmax=vmax,
        cmap=pd[varname]["cmap"],
    )

    cb = plt.colorbar(cs, ticks=ticks, spacing="uniform")
    cb.set_label("u [m s$^{-1}$]")

    ax.set_xlabel("x [km]")
    ax.set_ylabel("z [km]" if (z_km is not None and z_km.dtype != int) else "z level")
    ax.xaxis.set_minor_locator(MultipleLocator(50))

    # Time label (hours)
    # var.time is usually in seconds for each output slice; if missing, use frame index
    if hasattr(var, "time"):
        t_hours = np.asarray(var.time)[frame_ids] / 3600.0
    else:
        t_hours = frame_ids.astype(float)

    title = ax.set_title(f"Horizontal velocity (x,z), t = {t_hours[0]:.2f} h")

    # Update function: clear and redraw contourf (simple + robust)
    def update(i):
        ax.collections.clear()
        ax.contourf(
            x_km,
            z_km,
            U_anim[i, :, :],
            valRange,
            vmin=vmin,
            vmax=vmax,
            cmap=pd[varname]["cmap"],
        )
        title.set_text(f"Horizontal velocity (x,z), t = {t_hours[i]:.2f} h")
        return ax.collections + [title]

    ani = FuncAnimation(fig, update, frames=len(frame_ids), interval=50, blit=False)

    # Save MP4 via ffmpeg
    writer = FFMpegWriter(fps=int(args.fps))
    ani.save(args.outname, writer=writer, dpi=int(args.dpi))

    plt.show()


if __name__ == "__main__":
    main()