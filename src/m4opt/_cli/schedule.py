import shlex
import sys
from itertools import pairwise
from typing import Annotated

import numpy as np
import synphot
import typer
from astropy import units as u
from astropy.coordinates import ICRS, Distance, SkyCoord
from astropy.table import QTable, vstack
from astropy.time import Time
from astropy_healpix import HEALPix
from click import UsageError
from docplex.mp.progress import ProgressData, ProgressDataRecorder
from ligo.skymap import distance
from ligo.skymap.bayestar import rasterize
from ligo.skymap.io import read_sky_map
from scipy import stats

from time import perf_counter # for tracking planning time

from .. import __version__, missions
from ..dynamics import nominal_roll
from ..fov import footprint_healpix
from ..milp import Model
from ..observer import EarthFixedObserverLocation
from ..synphot import TabularScaleFactor, observing
from ..synphot.extinction import DustExtinction
from ..utils.console import progress, status
from ..utils.numpy import clump_nonzero_inclusive, full_indices
from .core import app


def invert_footprints(footprints, n_pixels):
    """
    Construct a map from pixel index to footprints containing those pixels.

    Examples
    --------
    >>> from m4opt._cli.schedule import invert_footprints
    >>> invert_footprints([[1, 2, 3], [0, 2, 3]], 4)
    [array([1]), array([0]), array([0, 1]), array([0, 1])]
    """
    pixels_to_fields_map = [[] for _ in range(n_pixels)]
    for i, js in enumerate(footprints):
        for j in js:
            pixels_to_fields_map[j].append(i)
    return [np.asarray(field_indices) for field_indices in pixels_to_fields_map]


def invert_footprints_to_regions(footprints, n_pixels):
    """
    Construct a map from pixel index to disjoint regions.

    Examples
    --------
    >>> from m4opt._cli.schedule import invert_footprints_to_regions
    >>> invert_footprints_to_regions([[1, 2, 3], [0, 2, 3]], 4)
    ([1, 0, 2, 2], [array([0]), array([1]), array([0, 1])])
    """
    pixels_to_fields_map = [
        tuple(field_indices)
        for field_indices in invert_footprints(footprints, n_pixels)
    ]
    region_to_fields_map = {
        footprint: i for i, footprint in enumerate(set(pixels_to_fields_map))
    }
    pixel_to_region_map = [
        region_to_fields_map[footprint] for footprint in pixels_to_fields_map
    ]
    region_to_fields_map = [
        np.asarray(fields, dtype=np.intp) for fields in region_to_fields_map
    ]
    return pixel_to_region_map, region_to_fields_map


LARGE_EXPTIME = 1e10


def prepare_piecewise_breakpoints(breakpoints):
    isinf_indices = np.flatnonzero(breakpoints[:, 1] >= LARGE_EXPTIME)
    if len(isinf_indices) > 0:
        breakpoints = breakpoints[: isinf_indices[0]]
    return [tuple(col.item() for col in row) for row in breakpoints]


@app.command()
@progress()
def schedule(
    skymap: Annotated[
        typer.FileBinaryRead,
        typer.Argument(help="Sky map filename", metavar="INPUT.multiorder.fits"),
    ],
    schedule: Annotated[
        typer.FileTextWrite,
        typer.Argument(
            help="Output filename for generated schedule", metavar="SCHEDULE.ecsv"
        ),
    ],
    exptime_min: Annotated[
        list[u.Quantity[u.physical.time]],
        typer.Option(
            help="Minimum exposure time for each observation. Repeat the "
            "option to give each bandpass its own exposure time, in the same "
            "order as --bandpass; a single value applies to every bandpass",
        ),
    ],
    mission: Annotated[
        missions.Mission, typer.Option(show_default="uvex")
    ] = missions.uvex,
    skygrid: Annotated[
        str | None,
        typer.Option(
            help="Name of sky grid to use, if the mission supports multiple sky grids.",
        ),
    ] = None,
    event_time: Annotated[
        Time | None,
        typer.Option(
            help="Time of the event, which --delay and --deadline are measured "
            "from. Defaults to the DATE-OBS field in the sky map header.",
        ),
    ] = None,
    delay: Annotated[
        u.Quantity,
        typer.Option(
            help="Delay from time of event until the start of observations",
        ),
    ] = 0 * u.day,
    deadline: Annotated[
        u.Quantity,
        typer.Option(
            help="Maximum time from event until the end of observations",
        ),
    ] = 1 * u.day,
    time_step: Annotated[
        u.Quantity,
        typer.Option(
            help="Time step for evaluating field of regard",
        ),
    ] = 1 * u.min,
    ###### added: make time window configurable
    time_windows: Annotated[
        bool,
        typer.Option(
            "--time-windows/--no-time-windows",
            help="Enforce mission field-of-regard visibility windows",
        ),
    ] = True,
    ####
    exptime_max: Annotated[
        u.Quantity,
        typer.Option(
            help="Maximum exposure time for each observation",
        ),
    ] = np.inf * u.s,
    absmag_mean: Annotated[
        float | None,
        typer.Option(
            help="Mean AB absolute magnitude of source",
            show_default="disable adaptive exposure time",
        ),
    ] = None,
    absmag_stdev: Annotated[
        float,
        typer.Option(
            help="Standard deviation of AB absolute magnitude of source",
            show_default="AB absolute magnitude is fixed at the value provided by --absmag-mean",
        ),
    ] = 0.0,
    appmag_dist: Annotated[
        bool, typer.Option(help="Enable point-wise distribution of apparent magnitude")
    ] = True,
    snr: Annotated[float, typer.Option(help="Signal to noise ratio for detection")] = 5,
    bandpass: Annotated[
        list[str] | None,
        typer.Option(
            help="Name of detector bandpass. Repeat the option to cycle through "
            "several bandpasses on successive visits; for example, "
            "--bandpass g --bandpass r observes each field in g and then in r. "
            "Visits are grouped into contiguous blocks of a single bandpass so "
            "that the filter is exchanged only between blocks."
        ),
    ] = None,
    visits: Annotated[int, typer.Option(min=1, help="Number of visits")] = 2,
    cadence: Annotated[
        u.Quantity,
        typer.Option(help="Minimum time separation between visits"),
    ] = 30 * u.min,
    nside: Annotated[int, typer.Option(help="HEALPix resolution")] = 512,
    max_fields: Annotated[
        int,
        typer.Option(
            min=1,
            help="Consider only this many of the most probable fields. Raising "
            "it grows the problem roughly quadratically",
        ),
    ] = 50,
    timelimit: Annotated[
        u.Quantity,
        typer.Option(
            help="Time limit for MILP solver",
            rich_help_panel="Solver Options",
        ),
    ] = 1e75 * u.s,
    memory: Annotated[
        u.Quantity,
        typer.Option(
            help="Maximum solver memory usage before terminating",
            rich_help_panel="Solver Options",
        ),
    ] = np.inf * u.GiB,
    jobs: Annotated[
        int,
        typer.Option(
            "--jobs",
            "-j",
            min=0,
            help="Number of threads for parallel processing, or 0 for all cores",
            rich_help_panel="Solver Options",
        ),
    ] = 0,
    ####### Daisy: for cpp solvers
    cpp_algorithm: Annotated[
        str | None,
        typer.Option(
            help="Use this no-window C++ algorithm instead of the MILP",
            rich_help_panel="Solver Options",
        ),
    ] = None,
    ########
    cutoff: Annotated[
        float | None,
        typer.Option(
            min=0,
            max=1,
            help="Objective cutoff. Give up if there are no feasible solutions with objective value greater than or equal to this value",
            rich_help_panel="Solver Options",
        ),
    ] = None,
    write_progress: Annotated[
        typer.FileTextWrite | None,
        typer.Option(
            help="Save a time series of the CPLEX objective value and best bound to this file",
            metavar="PROGRESS.ecsv",
            rich_help_panel="Solver Options",
        ),
    ] = None,
    write_model: Annotated[
        typer.FileBinaryWrite | None,
        typer.Option(
            help="Export the MILP model in LP, SAV, or MPS format. Mainly useful for troubleshooting purposes",
            metavar="MODEL.{lp,mps,sav}[.gz]",
            rich_help_panel="Solver Options",
        ),
    ] = None,
):
    """
    Generate an observing plan for a GW sky map.

    \b
    The scheduler has three modes:

    \b
    1. Fixed exposure time. Every field has the same exposure time given by the
       --exptime-min option. This mode is selected if you omit the
       --absmag-mean option.

    \b
    2. Variable exposure time. Each field may have a different exposure time,
       adjusted for the posterior median distance along each line of sight.
       This mode is selected if you specify a value for the --absmag-mean
       option but also pass the --no-appmag-dist option.

    \b
    3. Variable exposure time with an absolute magnitude distribution. Each
       field may have a different exposure time, adjusted to optimize the
       detection probability given the posterior distance distribution and a
       Gaussian distribution of absolute magnitudes. This mode is selected if
       you specify the --absmag-mean option (and, optionally, the
       --absmag-stdev option).

    \b
    Repeating the --bandpass option makes successive visits cycle through the
    bandpasses; for example, --bandpass g --bandpass r observes every field in g
    and then every field in r. Visits are grouped into contiguous blocks of a single
    bandpass, so that every field is observed for the kth time before any field
    is observed for the (k+1)th, and the filter is exchanged once per block
    boundary however many fields are observed.
    """
    ######## Daisy: added, start timer for planer
    planning_started = perf_counter()
    ########

    adaptive_exptime = absmag_mean is not None

    # Successive visits cycle through the requested bandpasses, so that
    # --bandpass g --bandpass r over three visits gives g, r, g.
    visit_bandpasses = [
        bandpass[i % len(bandpass)] if bandpass else None for i in range(visits)
    ]
    visit_exptime_min_s = u.Quantity(
        [exptime_min[i % len(exptime_min)] for i in range(visits)]
    ).to_value(u.s)
    if adaptive_exptime and bandpass is not None and len(bandpass) > 1:
        raise NotImplementedError(
            "A variable exposure time is not supported with more than one bandpass."
        )
    filter_changes = [lhs != rhs for lhs, rhs in pairwise(visit_bandpasses)]
    with status("loading sky map"):
        hpx = HEALPix(nside, frame=ICRS(), order="nested")
        skymap_moc = read_sky_map(skymap, moc=True)
        skymap_flat = rasterize(skymap_moc, hpx.level)

        print(skymap_moc.meta)

        if event_time is None:
            # The sky map carries the trigger time unless one was given.
            try:
                gps_time = skymap_moc.meta["gps_time"]
            except KeyError:
                raise UsageError(
                    f'The sky map "{skymap.name}" has no DATE-OBS in its header, '
                    "which is where the time of the event is read from. "
                    "Pass --event-time instead."
                ) from None
            event_time = Time(Time(gps_time, format="gps").utc, format="iso")

    with status("propagating orbit"):
        obstimes = event_time + np.arange(
            delay, deadline + time_step, time_step, like=time_step
        )
        observer_locations = mission.observer_location(obstimes)

    with status("evaluating field of regard"):
        if not isinstance(mission.skygrid, dict):
            target_coords = mission.skygrid
        elif skygrid in mission.skygrid:
            target_coords = mission.skygrid[skygrid]
        else:
            raise UsageError(
                f"skygrid '{skygrid}' not found. Options: {', '.join(map(str, mission.skygrid.keys()))}"
            )

        # The row of the grid is the mission's own name for the field. A
        # mission that numbers its fields leaves gaps, masked out of the grid
        # and dropped here so that everything below is dense.
        keep = ~target_coords.mask
        field_ids = np.arange(len(target_coords))[keep]
        target_coords = target_coords.unmasked[keep]
        # FIXME: https://github.com/astropy/astropy/issues/17030
        target_coords = SkyCoord(target_coords.ra, target_coords.dec)
        cadence_s = cadence.to_value(u.s)
        obstimes_s = (obstimes - obstimes[0]).to_value(u.s)

        ###### Daisy: commented out, orginal code with visiability window
        # observable_intervals = np.asarray(
        #     [
        #         obstimes_s[intervals]
        #         for intervals in clump_nonzero_inclusive(
        #             mission.constraints(
        #                 observer_locations,
        #                 target_coords[:, np.newaxis],
        #                 obstimes,
        #             )
        #         )
        #     ],
        #     dtype=object,
        # )
        ####### added, make time window configurable
        if time_windows:
            observable_intervals = np.asarray(
                [
                    obstimes_s[intervals]
                    for intervals in clump_nonzero_inclusive(
                        mission.constraints(
                            observer_locations,
                            target_coords[:, np.newaxis],
                            obstimes,
                        )
                    )
                ],
                dtype=object,
            )
        else:
            horizon_s = (deadline - delay).to_value(u.s)
            observable_intervals = np.empty(
                len(target_coords),
                dtype=object,
            )

            for field in range(len(target_coords)):
                observable_intervals[field] = np.asarray(
                    [[0.0, horizon_s]],
                    dtype=float,
                )
        ##########


        # Keep only intervals that are at least as long as the exposure time.
        exptime_min_s = visit_exptime_min_s.min()
        observable_intervals = np.asarray(
            [
                intervals[intervals[:, 1] - intervals[:, 0] >= exptime_min_s]
                for intervals in observable_intervals
            ],
            dtype=object,
        )

        # Discard fields that are not observable.
        good = np.asarray([len(intervals) > 0 for intervals in observable_intervals])
        observable_intervals = observable_intervals[good]
        target_coords = target_coords[good]
        field_ids = field_ids[good]

    with status("calculating footprints"):
        if isinstance(mission.observer_location, EarthFixedObserverLocation):
            rolls = np.zeros(len(target_coords)) * u.deg
        else:
            # Compute nominal roll angles for optimal solar power.
            # The nominal roll angle varies as a function of sky position and time.
            # We compute it for the start of the schedule because we assume that it
            # does not change much over the duration.
            rolls = nominal_roll(observer_locations[0], target_coords, event_time)
        footprints = footprint_healpix(hpx, mission.fov, target_coords, rolls)

        # Consider only the most probable fields.
        n_fields = max_fields #originally 50
        print("processing top", n_fields, "of", len(target_coords), "fields available")
        if len(target_coords) > n_fields:
            good = np.argpartition(
                [-skymap_flat[footprint]["PROB"].sum() for footprint in footprints],
                n_fields,
            )[:n_fields]
            target_coords = target_coords[good]
            rolls = rolls[good]
            footprints = footprints[good]
            observable_intervals = observable_intervals[good]
            field_ids = field_ids[good]
        else:
            n_fields = len(target_coords)

        # Throw away pixels that are not contained in any fields.
        good = (
            np.unique(np.concatenate(footprints))
            if len(footprints) > 0
            else np.asarray([], dtype=np.intp)
        )
        imap = np.empty(len(skymap_flat), dtype=np.intp)
        imap[good] = np.arange(len(good))
        skymap_flat = skymap_flat[good]
        footprints = np.asarray(
            [imap[footprint] for footprint in footprints], dtype=object
        )
        n_pixels = len(skymap_flat)

        if adaptive_exptime:
            pixel_to_region_map, region_to_fields_map = invert_footprints_to_regions(
                footprints, n_pixels
            )
            n_regions = len(region_to_fields_map)
        else:
            pixels_to_fields_map = invert_footprints(footprints, n_pixels)

    if adaptive_exptime:
        if mission.detector is None:
            raise NotImplementedError("This mission does not define a detector model")
        with status("evaluating exposure time map"):
            if appmag_dist:
                distmean, diststd, _ = distance.parameters_to_moments(
                    skymap_flat["DISTMU"],
                    skymap_flat["DISTSIGMA"],
                )
                logdist_sigma2 = np.log1p(np.square(diststd / distmean))
                logdist_sigma = np.sqrt(logdist_sigma2)
                logdist_mu = np.log(distmean) - 0.5 * logdist_sigma2
                a = 5 / np.log(10)
                appmag_mu = absmag_mean + a * logdist_mu + 25
                appmag_sigma = np.sqrt(
                    np.square(absmag_stdev) + np.square(a * logdist_sigma)
                )
                quantiles = np.linspace(0.05, 0.95, 5)
                appmag_quantiles = stats.norm(
                    loc=appmag_mu[:, np.newaxis], scale=appmag_sigma[:, np.newaxis]
                ).ppf(quantiles)
                # FIXME: prune pixels with infinite distance
                appmag_quantiles[np.isposinf(appmag_mu)] = np.inf

                with observing(
                    observer_location=observer_locations[0],
                    target_coord=hpx.healpix_to_skycoord(good)[:, np.newaxis],
                    obstime=obstimes[0],
                ):
                    exptime_pixel_s = mission.detector.get_exptime(
                        snr,
                        synphot.SourceSpectrum(synphot.ConstFlux1D(0 * u.ABmag))
                        * synphot.SpectralElement(
                            TabularScaleFactor(
                                (
                                    appmag_quantiles * u.mag(u.dimensionless_unscaled)
                                ).to_value(u.dimensionless_unscaled)
                            )
                        )
                        * DustExtinction(),
                        visit_bandpasses[0],
                    ).to_value(u.s)
                exptime_max_s = max(
                    min(
                        exptime_max.to_value(u.s),
                        deadline.to_value(u.s),
                    ),
                    exptime_min_s,
                )
                piecewise_breakpoints = np.pad(
                    np.stack(
                        (
                            np.tile(quantiles[np.newaxis, :], (len(skymap_flat), 1)),
                            exptime_pixel_s,
                        ),
                        axis=-1,
                    ),
                    [(0, 0), (1, 0), (0, 0)],
                )
            else:
                distmod = Distance(skymap_moc.meta["distmean"] * u.Mpc).distmod
                with observing(
                    observer_location=observer_locations[0],
                    target_coord=hpx.healpix_to_skycoord(good),
                    obstime=obstimes[0],
                ):
                    exptime_pixel_s = mission.detector.get_exptime(
                        snr,
                        synphot.SourceSpectrum(
                            synphot.ConstFlux1D(absmag_mean * u.ABmag + distmod)
                        )
                        * DustExtinction(),
                        visit_bandpasses[0],
                    ).to_value(u.s)
                    ######## Daisy: added, for debug and test purpose
                    ######## Print the distribution of P2 exposure times.
                    finite_exptime = exptime_pixel_s[
                        np.isfinite(exptime_pixel_s)
                    ]

                    if finite_exptime.size > 0:
                        print(
                            "Required exposure percentiles (seconds):",
                            np.percentile(
                                finite_exptime,
                                [0, 25, 50, 75, 90, 95, 99, 100],
                            ),
                        )
                    else:
                        print("No pixels have a finite required exposure time.")
                    #######
                exptime_min_s = min(
                    max(exptime_min_s, exptime_pixel_s.min(initial=exptime_min_s)),
                    exptime_max.to_value(u.s),
                )
                exptime_max_s = max(
                    min(
                        exptime_max.to_value(u.s),
                        deadline.to_value(u.s),
                        exptime_pixel_s.max(initial=exptime_max.to_value(u.s)),
                    ),
                    exptime_min_s,
                )

    with status("calculating slew times"):
        slew_i, slew_j = np.triu_indices(n_fields, 1)
        slew_time_s = mission.slew.time(
            target_coords[slew_i],
            target_coords[slew_j],
            rolls[slew_i],
            rolls[slew_j],
        ).to_value(u.s)

    ######## Daisy: added, Call one of the C++ solvers at this point
    ######## Everything before this point remains M4OPT's normal preprocessing:
    ######## (field selection, footprint generation, pixel compaction, exposure-time calculation, and slew-time calculation.)
    if cpp_algorithm is not None:
        if visits != 1:
            raise UsageError(
                "C++ solvers currently support --visits 1 only"
            )

        if time_windows:
            raise UsageError(
                "C++ solvers currently require --no-time-windows"
            )

        if adaptive_exptime and appmag_dist:
            raise UsageError(
                "C++ solvers support deterministic P2 only; "
                "add --no-appmag-dist"
            )

        if n_fields == 0:
            raise UsageError(
                "There are no observable fields to optimize"
            )

        # Import m4opt_solvers extension module.
        # Make sure the path for compiled .so(linux) file are exported
        try:
            import m4opt_solvers
        except ImportError as error:
            raise UsageError(
                "Could not import the m4opt_solvers C++ extension. "
                "Build the .so file and add its directory to PYTHONPATH."
            ) from error

        # Check the algorithm names.
        available_algorithms = m4opt_solvers.available_algorithms()

        normalized_algorithm = (
            cpp_algorithm.strip().lower().replace("-", "_").replace(" ", "_")
        )

        # Normalize algorithm names
        if normalized_algorithm == "ilp_continuous_time":
            normalized_algorithm = "ilp_continuous"

        if normalized_algorithm not in available_algorithms:
            raise UsageError(
                f"Unknown C++ algorithm {cpp_algorithm!r}. "
                f"Available algorithms: {', '.join(available_algorithms)}"
            )

        # M4OPT evaluates only the upper triangle of the slew-time matrix.
        # Reconstruct the complete symmetric matrix for the C++ solvers.
        # This matrix contains pure slew times. 
        slew_matrix_s = np.zeros(
            (n_fields, n_fields),
            dtype=float,
        )

        slew_matrix_s[slew_i, slew_j] = slew_time_s
        slew_matrix_s[slew_j, slew_i] = slew_time_s

        # Every pixel index in member_pixels refers to an entry in pixel_probabilities.
        pixel_probabilities = np.asarray(
            skymap_flat["PROB"],
            dtype=float,
        ).tolist()

        member_pixels = [
            np.asarray(
                footprint,
                dtype=np.int64,
            ).tolist()
            for footprint in footprints
        ]

        # The physical observing interval starts at obstimes[0], which already includes M4OPT's requested delay.
        budget_s = (deadline - delay).to_value(u.s)

        # A zero C++ time limit means unlimited. 
        # While M4OPT represents its default unlimited time limit using a very large quantity.
        solver_time_limit_s = timelimit.to_value(u.s)
        if not np.isfinite(solver_time_limit_s) or solver_time_limit_s >= 1e50:
            solver_time_limit_s = 0.0

        solver_call_started = perf_counter() # planing time counter
        with status(
            f"solving with C++ {normalized_algorithm}"
        ):
            if adaptive_exptime:
                # Deterministic variable-exposure P2.
                # exptime_pixel_s gives the exposure threshold required to detect each compact pixel. 
                result = m4opt_solvers.solve_p2(
                    slew_seconds=slew_matrix_s.tolist(),
                    pixel_probabilities=pixel_probabilities,
                    member_pixels=member_pixels,
                    required_exposure_seconds=np.asarray(
                        exptime_pixel_s,
                        dtype=float,
                    ).tolist(),
                    minimum_exposure_seconds=exptime_min_s,
                    maximum_exposure_seconds=exptime_max_s,
                    budget_seconds=budget_s,
                    algorithm=normalized_algorithm,
                    time_limit_seconds=solver_time_limit_s,
                    thread_count=jobs,
                )
            else:
                # Fixed-exposure P1. Every field receives the same exposure duration selected by M4OPT's --exptime-min option.
                dwell_seconds = np.full(
                    n_fields,
                    exptime_min_s,
                    dtype=float,
                ).tolist()

                result = m4opt_solvers.solve_p1(
                    slew_seconds=slew_matrix_s.tolist(),
                    pixel_probabilities=pixel_probabilities,
                    member_pixels=member_pixels,
                    dwell_seconds=dwell_seconds,
                    budget_seconds=budget_s,
                    algorithm=normalized_algorithm,
                    time_limit_seconds=solver_time_limit_s,
                    thread_count=jobs,
                )

        # Get planning time
        solver_call_seconds = (
            perf_counter() - solver_call_started
        )

        if not result["has_path"]:
            raise UsageError(
                f"C++ algorithm {normalized_algorithm!r} "
                "did not return a feasible path"
            )

        # The C++ result is already in chronological observation order.
        selected = np.asarray(
            result["tile_indices"],
            dtype=np.intp,
        )

        start_seconds = np.asarray(
            result["start_seconds"],
            dtype=float,
        )

        exposure_seconds = np.asarray(
            result["exposure_seconds"],
            dtype=float,
        )

        if not (
            len(selected)
            == len(start_seconds)
            == len(exposure_seconds)
        ):
            raise RuntimeError(
                "The C++ solver returned result arrays with different lengths"
            )

        if np.any(selected < 0) or np.any(selected >= n_fields):
            raise RuntimeError(
                "The C++ solver returned an invalid field index"
            )

        if len(np.unique(selected)) != len(selected):
            raise RuntimeError(
                "The C++ solver returned a repeated physical field"
            )

        if result["duration_seconds"] > budget_s + 1e-8:
            raise RuntimeError(
                "The C++ solver returned a schedule exceeding the deadline"
            )
        ## valid result from cpp solvers
        duration_seconds = float(result["duration_seconds"])
        tolerance = 1e-8

        if (
            not np.all(np.isfinite(start_seconds))
            or not np.all(np.isfinite(exposure_seconds))
            or not np.isfinite(duration_seconds)
        ):
            raise RuntimeError(
                "The C++ solver returned non-finite timing values"
            )

        if (
            np.any(start_seconds < 0)
            or np.any(exposure_seconds < 0)
            or duration_seconds < 0
        ):
            raise RuntimeError(
                "The C++ solver returned negative timing values"
            )

        if (
            len(start_seconds) > 1
            and np.any(np.diff(start_seconds) < -tolerance)
        ):
            raise RuntimeError(
                "The C++ observations are not in chronological order"
            )

        if np.any(
            start_seconds + exposure_seconds
            > budget_s + tolerance
        ):
            raise RuntimeError(
                "A C++ observation ends after the deadline"
            )

        if duration_seconds > budget_s + tolerance:
            raise RuntimeError(
                "The C++ solver returned a schedule exceeding the deadline"
            )
        ## Organize result in Qtable
        table = QTable(
            {
                "action": np.full(
                    len(selected),
                    "observe",
                ),
                "start_time": (
                    obstimes[0]
                    + start_seconds * u.s
                ),
                "duration": exposure_seconds * u.s,
                "target_coord": target_coords[selected],
                "roll": rolls[selected],
                "field_id": field_ids[selected],
                "bandpass": np.repeat(
                    np.array(
                        [visit_bandpasses[0] or ""],
                        dtype=str,
                    ),
                    len(selected),
                ),
            },
            descriptions={
                    "action": "Action for the spacecraft",
                    "start_time": "Start time of segment",
                    "duration": "Duration of segment",
                    "target_coord": "Coordinates of the center of the FOV",
                    "roll": "Position angle of the FOV",
                    "field_id": "The mission's ID for the field observed",
                    "bandpass": "Detector bandpass",
            },
            meta={
                "command": shlex.join(sys.argv),
                "version": __version__,
                "args": {
                    "deadline": deadline,
                    "delay": delay,
                    "mission": mission.name,
                    "skygrid": skygrid,
                    "nside": nside,
                    "max_fields": max_fields,
                    "time_step": time_step,
                    "skymap": skymap.name,
                    "event_time": event_time.isot,
                    "visits": visits,
                    "exptime_min": exptime_min,
                    "exptime_max": exptime_max,
                    "absmag_mean": absmag_mean,
                    "absmag_stdev": absmag_stdev,
                    "appmag_dist": appmag_dist,
                    "bandpass": visit_bandpasses,
                    "snr": snr,
                    "cutoff": cutoff,
                    "time_windows": time_windows,
                    "solution_time": result["runtime_seconds"] * u.s,
                    "solver_call_time": solver_call_seconds * u.s,
                },
                "objective_value": result["coverage"],
                "best_bound": result["best_bound"],
                "solution_status": (
                    "time limit"
                    if result["timed_out"]
                    else "ok"
                ),
                "solution_time": (
                    result["runtime_seconds"] * u.s
                ),
                "cpp_algorithm": result["algorithm"],
                "cpp_no_time_windows": True,
            },
        )

        table.sort("start_time")
        table.meta["planning_time"] = (
            perf_counter() - planning_started
        ) * u.s

        table.write(
            schedule,
            format="ascii.ecsv",
            overwrite=True,
        )

        return
    ##########

     # Run original M4OPT MILP solver from here
    with Model(
        timelimit=timelimit, jobs=jobs, memory=memory, lowercutoff=cutoff
    ) as model:
        #### Daisy: define optimality GAP
        model.context.cplex_parameters.mip.tolerances.mipgap = 0.001

        with status("assembling MILP model"):
            if adaptive_exptime and appmag_dist:
                pixel_vars = model.continuous_vars(
                    n_pixels,
                    lb=0,
                    ub=[
                        breakpoints[(breakpoints[:, 1] < LARGE_EXPTIME), 0].max()
                        for breakpoints in piecewise_breakpoints
                    ],
                )
            else:
                pixel_vars = model.binary_vars(n_pixels)
            field_vars = model.binary_vars(n_fields)
            time_field_visit_vars = model.continuous_vars(
                (n_fields, visits),
            )
            if adaptive_exptime:
                exptime_field_vars = (
                    model.semicontinuous_vars
                    if exptime_min_s > 0
                    else model.continuous_vars
                )(n_fields, lb=exptime_min_s, ub=exptime_max_s)
                exptime_region_vars = model.continuous_vars(n_regions)

            # Add constraints on observability windows for each field
            with status("adding field of regard constraints"):
                for time_visit_vars, exptime, intervals in zip(
                    time_field_visit_vars,
                    exptime_field_vars
                    if adaptive_exptime
                    else np.tile(visit_exptime_min_s, (n_fields, 1)),
                    observable_intervals,
                ):
                    assert len(intervals) > 0
                    begin, end = intervals.T
                    if len(intervals) == 1:
                        model.add_constraints_(
                            time_visit_vars - begin - 0.5 * exptime >= 0
                        )
                        model.add_constraints_(
                            time_visit_vars - end + 0.5 * exptime <= 0
                        )
                    else:
                        visit_interval_vars = model.binary_vars(
                            (visits, len(intervals))
                        )
                        for interval_vars in visit_interval_vars:
                            model.add_constraint_(
                                model.sum_vars_all_different(interval_vars) >= 1
                            )
                        model.add_indicators(
                            visit_interval_vars,
                            time_visit_vars[:, np.newaxis] - begin - 0.5 * exptime >= 0,
                        )
                        model.add_indicators(
                            visit_interval_vars,
                            time_visit_vars[:, np.newaxis] - end + 0.5 * exptime <= 0,
                        )

            # Two observations are separated by half of each of their exposure
            # times, so a pair drawn from consecutive visits is separated by the
            # mean of theirs. Both the cadence and the slew constraints below
            # measure that separation.
            mean_consecutive_exptime_s = 0.5 * (
                visit_exptime_min_s[:-1] + visit_exptime_min_s[1:]
            )

            if visits > 1:
                with status("adding cadence constraints"):
                    if adaptive_exptime:
                        rhs = (cadence_s * field_vars + exptime_field_vars)[
                            :, np.newaxis
                        ]
                    else:
                        rhs = np.multiply.outer(
                            field_vars, cadence_s + mean_consecutive_exptime_s
                        )
                    model.add_constraints_(
                        (time_field_visit_vars[:, 1:] - time_field_visit_vars[:, :-1])
                        >= rhs
                    )

            with status("adding slew constraints"):
                # Zero or less unless both fields are observed, which relaxes
                # the constraint away for any pair that is not.
                both_observed = field_vars[slew_i] + field_vars[slew_j] - 1
                if adaptive_exptime:
                    rhs = (
                        0.5 * (exptime_field_vars[slew_i] + exptime_field_vars[slew_j])
                        + slew_time_s * both_observed
                    )
                    rhs_within = rhs
                    rhs_after = rhs
                else:
                    # Two observations also clear each other by the slew itself.
                    def _spacing(exptimes):
                        return (
                            slew_time_s[np.newaxis, :] + exptimes[:, np.newaxis]
                        ) * both_observed[np.newaxis, :]

                    rhs = rhs_within = _spacing(visit_exptime_min_s)
                    rhs_after = _spacing(mean_consecutive_exptime_s)

                if any(filter_changes):
                    # Every field is visited for the kth time before any field
                    # is visited for the k+1th, so the filter is exchanged once
                    # per block boundary however many fields are observed. The
                    # ordering also makes the absolute value redundant across
                    # visits, leaving it only within one.
                    exchange_s = mission.filter_exchange_time.to_value(u.s)
                    gap = (
                        rhs_after
                        + exchange_s * np.asarray(filter_changes)[:, np.newaxis]
                    )
                    within_visit = (
                        time_field_visit_vars[slew_i, :]
                        - time_field_visit_vars[slew_j, :]
                    )
                    after_i = (
                        time_field_visit_vars[slew_i, 1:]
                        - time_field_visit_vars[slew_j, :-1]
                    )
                    after_j = (
                        time_field_visit_vars[slew_j, 1:]
                        - time_field_visit_vars[slew_i, :-1]
                    )
                    model.add_constraints_(
                        model.abs(np.transpose(within_visit)) >= rhs_within
                    )
                    model.add_constraints_(np.transpose(after_i) >= gap)
                    model.add_constraints_(np.transpose(after_j) >= gap)
                else:
                    p, q = full_indices(visits)
                    if not adaptive_exptime:
                        rhs = _spacing(
                            0.5 * (visit_exptime_min_s[p] + visit_exptime_min_s[q])
                        )
                    model.add_constraints_(
                        model.abs(
                            time_field_visit_vars[slew_i, p[:, np.newaxis]]
                            - time_field_visit_vars[slew_j, q[:, np.newaxis]]
                        )
                        >= rhs
                    )

            if adaptive_exptime:
                with status("adding exposure time constraints"):
                    model.add_constraints_(
                        exptime_max_s * field_vars >= exptime_field_vars
                    )

            with status("adding coverage constraints"):
                if adaptive_exptime:
                    if appmag_dist:
                        for pixel_var, region_index, breakpoints in zip(
                            pixel_vars, pixel_to_region_map, piecewise_breakpoints
                        ):
                            breakpoints = prepare_piecewise_breakpoints(breakpoints)
                            if len(breakpoints) <= 1:
                                assert pixel_var.ub == 0
                            else:
                                model.add_constraint_(
                                    exptime_region_vars[region_index]
                                    >= model.piecewise(0, breakpoints, 0)(pixel_var)
                                )
                    else:
                        model.add_indicators(
                            pixel_vars,
                            [
                                exptime_region_vars[region] >= exptime_s
                                for region, exptime_s in zip(
                                    pixel_to_region_map, exptime_pixel_s
                                )
                            ],
                        )
                    model.add_constraints_(
                        [
                            model.max(*exptime_field_vars[field_indices]).item()
                            >= exptime_var
                            for field_indices, exptime_var in zip(
                                region_to_fields_map, exptime_region_vars
                            )
                        ]
                    )
                else:
                    model.add_constraints_(
                        pixel_vars
                        <= [
                            model.sum_vars_all_different(field_vars[field_indices])
                            for field_indices in pixels_to_fields_map
                        ]
                    )

            with status("adding cuts"):
                model.add_user_cut_constraint(
                    model.sum_vars_all_different(field_vars)
                    <= (deadline - delay).to_value(u.s) / visit_exptime_min_s.sum()
                )
                if adaptive_exptime:
                    model.add_user_cut_constraint(
                        model.sum_vars_all_different(exptime_field_vars)
                        <= (deadline - delay).to_value(u.s) / visits
                    )

            with status("adding objective function"):
                model.maximize(
                    model.scal_prod_vars_all_different(pixel_vars, skymap_flat["PROB"])
                )
        ###### Daisy: added, init timer for original solver
        solver_call_seconds = 0.0 
        ######
        if has_model := (
            model.number_of_constraints + model.objective_expr.number_of_terms() > 0
        ):
            with status("solving MILP model"):
                if write_progress is not None:
                    model.add_progress_listener(recorder := ProgressDataRecorder())

                if write_model is not None:
                    model.to_stream(write_model)
                solver_call_started = perf_counter() # Daisy: added, get M4OPT MILP planning time
                solution = model.solve()
                solver_call_seconds = (
                    perf_counter() - solver_call_started 
                )
        else:
            solution = None

        with status("writing results"):
            if write_progress is not None:
                QTable(
                    rows=recorder.recorded,
                    names=ProgressData._fields,
                    dtype=[int, bool, float, float, float, int, int, int, float, float],
                ).write(write_progress, format="ascii.ecsv", overwrite=True)

            if solution is None:
                field_values = np.zeros(field_vars.shape, dtype=bool)
                time_field_visit_values = np.empty(time_field_visit_vars.shape)
                exptime_field_values = np.empty(time_field_visit_vars.shape)
                objective_value = 0.0
            else:
                field_values = solution.get_values(field_vars) >= 0.5
                time_field_visit_values = solution.get_values(time_field_visit_vars)
                if adaptive_exptime:
                    exptime_per_field = solution.get_values(exptime_field_vars)
                    field_values &= exptime_per_field > 0
                    exptime_field_values = np.tile(
                        exptime_per_field[:, np.newaxis], visits
                    )
                else:
                    exptime_field_values = np.tile(visit_exptime_min_s, (n_fields, 1))
                objective_value = solution.get_objective_value()

            table = QTable(
                {
                    "action": np.full(field_values.sum() * visits, "observe"),
                    "start_time": obstimes[0]
                    + (
                        time_field_visit_values[field_values]
                        - 0.5 * exptime_field_values[field_values]
                    ).ravel()
                    * u.s,
                    "duration": exptime_field_values[field_values].ravel() * u.s,
                    "target_coord": target_coords[
                        np.tile(np.flatnonzero(field_values)[:, np.newaxis], visits)
                    ].ravel(),
                    "roll": rolls[
                        np.tile(np.flatnonzero(field_values)[:, np.newaxis], visits)
                    ].ravel(),
                    "field_id": field_ids[
                        np.tile(np.flatnonzero(field_values)[:, np.newaxis], visits)
                    ].ravel(),
                    "bandpass": np.tile(
                        np.array([band or "" for band in visit_bandpasses], dtype=str),
                        field_values.sum(),
                    ),
                },
                descriptions={
                    "action": "Action for the spacecraft",
                    "start_time": "Start time of segment",
                    "duration": "Duration of segment",
                    "target_coord": "Coordinates of the center of the FOV",
                    "roll": "Position angle of the FOV",
                    "field_id": "The mission's ID for the field observed",
                    "bandpass": "Detector bandpass",
                },
                meta={
                    "command": shlex.join(sys.argv),
                    "version": __version__,
                    "args": {
                        "deadline": deadline,
                        "delay": delay,
                        "mission": mission.name,
                        "skygrid": skygrid,
                        "nside": nside,
                        "max_fields": max_fields,
                        "time_step": time_step,
                        "skymap": skymap.name,
                        "event_time": event_time.isot,
                        "visits": visits,
                        "exptime_min": exptime_min,
                        "exptime_max": exptime_max,
                        "absmag_mean": absmag_mean,
                        "absmag_stdev": absmag_stdev,
                        "appmag_dist": appmag_dist,
                        "bandpass": visit_bandpasses,
                        "snr": snr,
                        "cutoff": cutoff,
                        ###### Daisy: added, add planning time to result for analysis
                        "time_windows": time_windows,
                        "solution_time": (
                            model.solve_details.time if has_model else 0
                        ) * u.s,
                        "solver_call_time": solver_call_seconds * u.s,
                        ######
                    },
                    "objective_value": objective_value,
                    "best_bound": model.best_bound if has_model else 0,
                    "solution_status": model.solve_details.status
                    if has_model
                    else "infeasible, no observable fields or pixels",
                    "solution_time": (model.solve_details.time if has_model else 0)
                    * u.s,
                },
            )
            table.sort("start_time")

            # Add orbit to table
            table.add_column(
                mission.observer_location(table["start_time"]),
                index=3,
                name="observer_location",
            )
            table["observer_location"].info.description = "Position of the spacecraft"

            # Add slew segments to table.
            if len(table) > 0:
                nrows = len(table) - 1
                slew_duration = mission.slew.time(
                    table["target_coord"][:-1],
                    table["target_coord"][1:],
                    table["roll"][:-1],
                    table["roll"][1:],
                )
                # The filter is exchanged while the telescope slews, so a
                # change costs only the excess over the slew itself.
                changed = table["bandpass"][:-1] != table["bandpass"][1:]
                slew_duration[changed] = np.maximum(
                    slew_duration[changed], mission.filter_exchange_time
                )
                slew_table = QTable(
                    {
                        "action": np.full(nrows, "slew"),
                        "start_time": (table["start_time"] + table["duration"])[:-1],
                        "duration": slew_duration,
                        "bandpass": np.full(nrows, ""),
                    }
                )
                table = vstack(
                    (
                        table,
                        slew_table,
                    )
                )

            table.sort("start_time")

            # Calculate total time spent observing, slewing, etc.,
            # as well as the amount of unused slack time
            total_time_by_action = (
                table["action", "duration"].group_by("action").groups.aggregate(np.sum)
            )
            table.meta["total_time"] = {
                str(row["action"]): row["duration"].to(u.s)
                for row in total_time_by_action
            }
            table.meta["total_time"]["slack"] = (
                deadline - delay - total_time_by_action["duration"].sum()
            ).to(u.s)
            # Daisy: add planning time as a section
            table.meta["planning_time"] = (
                perf_counter() - planning_started
            ) * u.s

            table.write(schedule, format="ascii.ecsv", overwrite=True)
