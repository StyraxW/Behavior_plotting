"""Fiber photometry loading, and its alignment to Bpod behavior.

The behavior side is exactly the same format lick_plotting.py already handles,
so file discovery, session loading and the trial-level accessors are imported
from there rather than repeated. What is new here is the photometry file and
the clock join between the two.

  Photometry ``<animal>_<yyyymmdd>_photometry_<NNNN>_Photodata.mat`` -> ``PhotoData``
             One continuous recording. Times are seconds from the start of
             acquisition, which begins before Bpod does.

The two clocks meet at ``PhotoData.IOsync``, a 1 kHz digital line that Bpod
drives HIGH for the duration of each trial's ITI state -- one rising edge per
trial start, one falling edge at cue onset.

:func:`align` turns those edges into a clock map in two parts. A **global fit**
through all the edges supplies the rate (the clocks differ by ~1e-5, several ms
across a session). Then **each trial is anchored to its own rising edge**, so a
trial-relative time ``rel`` lands at ``anchors[i] + rel * slope``. Per-trial
anchoring is the default: a trial's events can never drift away from its own
sync pulse, and a bad or missing edge stays confined to that trial instead of
tilting a line fitted through everything.

Pass ``anchor="fit"`` to any of ``event_times`` / ``lick_times`` / ``epoch`` --
or set ``sess.anchor`` -- to use the global line instead. On a clean recording
the two agree to well under a millisecond; the fit is marginally smoother
(it averages the +/-0.5 ms quantisation of 100 edges), while anchoring is more
robust to a bad edge. Neither difference is visible in a 30 Hz signal.

Typical use::

    import photometry_data as pdm

    sessions = pdm.load_all(pdm.BEHAVIOR_ROOT, pdm.PHOTOMETRY_ROOT)
    sess = sessions["SW007_20260905"]

    sess.trials                              # tidy per-trial DataFrame
    t, M = sess.epoch("Reward_on", (-2, 6))  # M is (n_trials, n_samples)
    M = pdm.baseline_subtract(M, t, (-2, -0.2))

lick_plotting's block-structure helpers (``infer_blocks``, ``describe_blocks``)
take a raw SessionData, so they work directly on ``sess.behavior``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

from lick_plotting import (
    _times,
    cue_per_trial,
    find_lick_event,
    find_session_files,
    load_session,
    n_trials,
    rewarded_per_trial,
)

# <animal>_<yyyymmdd>_photometry_<run>_Photodata.mat
PHOTO_RE = re.compile(
    r"^(?P<animal>[^_]+)_(?P<date>\d{8})_photometry_(?P<run>\d+)_Photodata$",
    re.IGNORECASE,
)
# <animal>_<protocol...>_<yyyymmdd>_<HHMMSS>.mat
BEHAVIOR_RE = re.compile(
    r"^(?P<animal>[^_]+)_(?P<protocol>.+)_(?P<date>\d{8})_(?P<time>\d{6})$"
)

# Defaults for running this file directly (see __main__). Analysis notebooks
# should define their own roots and pass them in, so the data source is visible
# where the analysis lives rather than buried in the library.
BEHAVIOR_ROOT = r"D:\WXC-related\Harvard\Uchida lab\Behavior_data"
PHOTOMETRY_ROOT = r"D:\WXC-related\Harvard\Uchida lab\Photometry_data"


# --------------------------------------------------------------------------
# Photometry
# --------------------------------------------------------------------------

@dataclass
class Photometry:
    """One continuous photometry recording (``PhotoData``).

    t : (n,) seconds from acquisition start, on a uniform 1 kHz grid.
    F : (n,) signal channel (470 nm), already normalised by the acquisition
        software -- values sit near 0, so treat it as dF/F, not raw counts.
    iso : (n,) isosbestic channel (405 nm), on the same grid.
    F_subtr : (n,) the file's motion-corrected signal. NOTE: in every file
        checked so far this is bit-identical to ``F``, i.e. no isosbestic
        subtraction was actually applied upstream. Use :func:`regress_iso`.
    sig, t_sig, iso_raw, t_iso : the un-interpolated ~30.3 Hz acquisition in
        raw detector units, from ``dat_orig``. The channels are interleaved,
        so their timestamps sit half a frame apart.
    sync : (n,) the Bpod digital line, on the same grid as ``t``. HIGH for the
        duration of each trial's ITI. The file stores a few more sync samples
        than signal samples; the tail is trimmed on load.
    """

    t: np.ndarray
    F: np.ndarray
    iso: np.ndarray
    F_subtr: np.ndarray
    sync: np.ndarray
    sync_name: str
    sig: np.ndarray
    t_sig: np.ndarray
    iso_raw: np.ndarray
    t_iso: np.ndarray
    path: Path | None = None
    animal: str | None = None
    date: str | None = None
    run: str | None = None

    @property
    def fs(self) -> float:
        """Sampling rate of the interpolated grid, in Hz."""
        return 1.0 / float(np.median(np.diff(self.t)))

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0])

    def _edges(self, direction: int) -> np.ndarray:
        idx = np.flatnonzero(np.diff(self.sync.astype(np.int16)) == direction)
        return self.t[idx + 1]

    @property
    def sync_rises(self) -> np.ndarray:
        """Photometry-clock times of each rising edge (trial starts)."""
        return self._edges(1)

    @property
    def sync_falls(self) -> np.ndarray:
        """Photometry-clock times of each falling edge (ITI end = cue onset)."""
        return self._edges(-1)

    @property
    def pulse_widths(self) -> np.ndarray:
        """Seconds the line stayed HIGH after each rising edge, NaN if never.

        One per rising edge, each paired with the first falling edge after it.
        Every pulse is a trial's ITI, so these all sit near the ITI duration --
        a much shorter one is the line being toggled by something that is not a
        trial, which is what :meth:`drop_leading_pulses` is for.
        """
        r, f = self.sync_rises, self.sync_falls
        j = np.searchsorted(f, r)
        return np.where(j < f.size, f[np.clip(j, 0, f.size - 1)] - r, np.nan)

    def drop_leading_pulses(self, n: int) -> "Photometry":
        """A copy with the first ``n`` sync pulses erased from the line.

        For pulses that are not trials -- the line toggled while the session was
        being set up, which shows up as a HIGH far shorter than an ITI. Erasing
        them on the line itself, rather than at each call site, is what keeps
        every consumer consistent: :func:`align`, ``sync_rises``/``sync_falls``
        and any TTL-derived event times then see the same edges::

            photo = load_photometry(path, skip_pulses=1)   # or .drop_leading_pulses(1)
        """
        if n <= 0:
            return self
        rises = np.flatnonzero(np.diff(self.sync.astype(np.int16)) == 1) + 1
        if n >= rises.size:
            raise ValueError(f"asked to drop {n} leading pulses but the line has "
                             f"only {rises.size}")
        sync = self.sync.copy()
        sync[:rises[n]] = 0     # everything up to the (n+1)-th rising edge
        return replace(self, sync=sync)

    def index_at(self, times) -> np.ndarray:
        """Nearest sample index for each photometry-clock time.

        The grid is uniform, so this is arithmetic rather than a search.
        Indices are clipped into range, so times outside the recording map to
        the first or last sample.
        """
        times = np.atleast_1d(np.asarray(times, dtype=float))
        i = np.rint((times - self.t[0]) * self.fs)
        return np.clip(i, 0, self.t.size - 1).astype(np.int64)

    def __repr__(self):
        who = self.animal or (self.path.stem if self.path else "?")
        return (f"<Photometry {who} {self.date or ''} "
                f"{self.duration:.0f}s @ {self.fs:.0f}Hz, "
                f"{self.sync_rises.size} sync pulses>")


def load_photometry(path, skip_pulses: int = 0) -> Photometry:
    """Load one ``*_Photodata.mat`` file.

    skip_pulses  drop this many leading sync pulses, for a line that was toggled
                 before the first real trial (see
                 :meth:`Photometry.drop_leading_pulses`).
    """
    path = Path(path)
    mat = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    if "PhotoData" not in mat:
        raise ValueError(f"no PhotoData in {path.name}")
    p = mat["PhotoData"]

    t = np.asarray(p.t, dtype=float).ravel()
    sync = np.asarray(p.IOsync.Data).ravel()
    # The sync vector runs a little past the end of the signal grid. Trim so
    # sample i of sync corresponds to t[i]; pad in the (unseen) other case.
    if sync.size > t.size:
        sync = sync[:t.size]
    elif sync.size < t.size:
        sync = np.concatenate([sync, np.zeros(t.size - sync.size, sync.dtype)])

    meta = PHOTO_RE.match(path.stem)
    photo = Photometry(
        t=t,
        F=np.asarray(p.F, dtype=float).ravel(),
        iso=np.asarray(p.iso, dtype=float).ravel(),
        F_subtr=np.asarray(p.F_subtr, dtype=float).ravel(),
        sync=sync.astype(np.int8),
        sync_name=str(p.IOsync.Name),
        sig=np.asarray(p.dat_orig.sig, dtype=float).ravel(),
        t_sig=np.asarray(p.dat_orig.t_sig, dtype=float).ravel(),
        iso_raw=np.asarray(p.dat_orig.iso, dtype=float).ravel(),
        t_iso=np.asarray(p.dat_orig.t_iso, dtype=float).ravel(),
        path=path,
        **({k: meta[k] for k in ("animal", "date", "run")} if meta else {}),
    )
    return photo.drop_leading_pulses(skip_pulses)


def regress_iso(F, iso):
    """Isosbestic-corrected signal: F minus its least-squares fit to iso.

    The file's own ``F_subtr`` is a copy of ``F``, so if you want motion
    correction you have to do it here. Returns F - (a*iso + b), which leaves
    the result centred on zero.
    """
    F, iso = np.asarray(F, dtype=float), np.asarray(iso, dtype=float)
    good = np.isfinite(F) & np.isfinite(iso)
    a, b = np.polyfit(iso[good], F[good], 1)
    return F - (a * iso + b)


def signal_of(photo: Photometry, which: str = "F") -> np.ndarray:
    """A whole-session trace: "F", "iso", "F_subtr", or "F_iso_regressed"."""
    if which == "F_iso_regressed":
        return regress_iso(photo.F, photo.iso)
    if which not in ("F", "iso", "F_subtr"):
        raise ValueError(f"unknown signal {which!r}")
    return getattr(photo, which)


def epoch_signal(photo: Photometry, times, window=(-2.0, 5.0), which="F",
                 fill=np.nan):
    """Peri-event signal matrix, from photometry-clock times alone.

    The behavior-free half of :meth:`Session.epoch`, which delegates here once
    it has turned an event name into times. Use it directly on a recording with
    **no behavior file**, where the events have to come from the sync line
    itself -- ``photo.sync_rises`` for trial starts, ``photo.sync_falls`` for
    cue onset, and a fixed protocol offset from there.

    Returns (t, M): ``t`` is (n_samples,) seconds relative to the event, ``M``
    is (n_events, n_samples). Rows whose time is NaN are all ``fill``, as are
    samples falling outside the recording.
    """
    ev = np.atleast_1d(np.asarray(times, dtype=float))
    y = signal_of(photo, which)
    step = 1.0 / photo.fs
    # Offsets in samples, so every row shares one time base exactly.
    offsets = np.arange(int(np.floor(window[0] / step)),
                        int(np.ceil(window[1] / step)) + 1)

    M = np.full((ev.size, offsets.size), float(fill))
    ok = np.isfinite(ev)
    if ok.any():
        idx = photo.index_at(ev[ok])[:, None] + offsets[None, :]
        valid = (idx >= 0) & (idx < y.size)
        rows = M[ok]
        rows[valid] = y[idx[valid]]
        M[ok] = rows
    return offsets * step, M


# --------------------------------------------------------------------------
# Trial table
# --------------------------------------------------------------------------

def state_names(session_data) -> list[str]:
    """Every state name appearing anywhere in the session, in file order.

    Reads ``_fieldnames`` rather than lick_plotting's ``_fields``, which
    returns a set and so would scramble the column order.
    """
    seen = {}
    for tr in np.atleast_1d(session_data.RawEvents.Trial):
        seen.update(dict.fromkeys(getattr(tr.States, "_fieldnames", ())))
    return list(seen)


def _window(states, name):
    """(onset, offset) of a state's first visit; (nan, nan) if never visited.

    A state visited more than once gives Bpod an Nx2 array, hence the ravel:
    we take the first visit.
    """
    t = np.ravel(_times(states, name))
    return (t[0], t[1]) if t.size >= 2 else (np.nan, np.nan)


def _per_trial(session_data, name, n):
    """A per-trial SessionData field as a length-n float array.

    Handles the cell-array fields (RewardDelivered) as well as plain numeric
    ones, and pads with NaN if the field is short or missing.
    """
    v = getattr(session_data, name, None)
    if v is None:
        return np.full(n, np.nan)
    v = np.atleast_1d(np.asarray(v)).ravel()
    if v.dtype == object:
        v = np.array([np.ravel(x)[0] if np.size(x) else np.nan for x in v])
    v = v.astype(float)
    return np.concatenate([v, np.full(max(0, n - v.size), np.nan)])[:n]


def trial_table(session_data, lick_event=None) -> pd.DataFrame:
    """One row per trial, with state timings flattened into columns.

    trial                 1-based trial number, as in MATLAB.
    trial_start/_end      Bpod session clock, seconds.
    trial_type, stim, block_transition, iti_dur, trace_dur, reward_ul
                          Per-trial session fields (``stim`` is StimTrials).
    <State>_on/_off       Onset/offset of every state, in **trial-relative**
                          seconds; NaN where the trial did not visit it.
    <State>_entered       Bool, convenience for the NaN check above.
    cue, rewarded         From lick_plotting's cue_per_trial/rewarded_per_trial.
    licks, n_licks        Lick times (object column), trial-relative.

    Trial-relative times are what Bpod stores. Add ``trial_start`` for session
    clock, or use :meth:`Session.event_times`, which also maps to photometry.
    """
    trials = np.atleast_1d(session_data.RawEvents.Trial)
    n = n_trials(session_data)
    if lick_event is None:
        try:
            lick_event = find_lick_event(session_data)
        except ValueError:
            lick_event = None

    df = pd.DataFrame({
        "trial": np.arange(1, n + 1),
        **{col: _per_trial(session_data, field, n) for col, field in (
            ("trial_start", "TrialStartTimestamp"),
            ("trial_end", "TrialEndTimestamp"),
            ("trial_type", "TrialType"),
            ("stim", "StimTrials"),
            ("block_transition", "BlockTransition"),
            ("iti_dur", "ITIDuration"),
            ("trace_dur", "TraceDuration"),
            ("reward_ul", "RewardDelivered"),
        )},
    })

    for s in state_names(session_data):
        w = np.array([_window(trials[i].States, s) for i in range(n)])
        df[f"{s}_on"], df[f"{s}_off"] = w[:, 0], w[:, 1]
        df[f"{s}_entered"] = ~np.isnan(w[:, 0])

    df["cue"] = cue_per_trial(session_data)
    df["rewarded"] = rewarded_per_trial(session_data)
    df["licks"] = [_times(trials[i].Events, lick_event) if lick_event
                   else np.array([]) for i in range(n)]
    df["n_licks"] = df["licks"].map(len)
    return df


def count_licks_in(df: pd.DataFrame, start_col: str, stop_col: str) -> np.ndarray:
    """Licks falling inside [start_col, stop_col] on each trial.

    Both columns are trial-relative, as are the stored lick times, so no clock
    conversion is needed::

        df["cue_licks"] = count_licks_in(df, "Cue1_on", "Cue1_off")
    """
    return np.array([
        np.count_nonzero((lk >= a) & (lk <= b))
        if np.size(lk) and np.isfinite(a) and np.isfinite(b) else 0
        for lk, a, b in zip(df["licks"], df[start_col], df[stop_col])
    ])


# --------------------------------------------------------------------------
# Clock alignment
# --------------------------------------------------------------------------

@dataclass
class Alignment:
    """Bpod -> photometry clock map, in two flavours.

    **Global fit** (``to_photometry``): ``photometry_t = slope * bpod_t +
    intercept``, one line through every sync edge. The slope is near 1 but not
    exactly 1 -- the acquisition clocks drift by ~1e-5 relative, several
    milliseconds across a session.

    **Per-trial anchoring** (``anchored``): each trial's events hang off that
    trial's own rising edge, so a trial-relative time ``rel`` lands at
    ``anchors[i] + rel * slope``. No error accumulates across the session and a
    single bad edge stays local to its own trial. This is what :class:`Session`
    uses by default.

    ``anchors[i]`` is the photometry-clock time of trial i's rising edge, or NaN
    if no edge could be assigned -- those trials fall back to the global fit.
    """

    slope: float
    intercept: float
    n_pulses: int
    n_trials: int
    residuals: np.ndarray
    anchors: np.ndarray = field(default_factory=lambda: np.array([]))
    trial_starts: np.ndarray = field(default_factory=lambda: np.array([]))

    def to_photometry(self, bpod_t):
        """Global fit: Bpod session-clock times -> photometry clock."""
        return self.slope * np.asarray(bpod_t, dtype=float) + self.intercept

    def to_bpod(self, photo_t):
        return (np.asarray(photo_t, dtype=float) - self.intercept) / self.slope

    def anchored(self, rel, idx=None):
        """Trial-relative times -> photometry clock, each on its own edge.

        ``rel`` holds one time per trial (NaN where a state was not visited);
        ``idx`` optionally selects a subset of trials. Trials without a usable
        edge fall back to the global fit, so the result is never NaN merely
        because an edge was missing.
        """
        rel = np.atleast_1d(np.asarray(rel, dtype=float))
        a = self.anchors if idx is None else self.anchors[idx]
        ts = self.trial_starts if idx is None else self.trial_starts[idx]
        if a.size != rel.size:
            raise ValueError(f"anchored() expects one time per trial: got "
                             f"{rel.size} times for {a.size} trials")
        out = a + rel * self.slope
        missing = ~np.isfinite(a)
        if missing.any():
            out[missing] = self.to_photometry(ts[missing] + rel[missing])
        return out

    @property
    def n_anchored(self) -> int:
        """Trials that got their own rising edge."""
        return int(np.count_nonzero(np.isfinite(self.anchors)))

    @property
    def max_residual(self) -> float:
        """Worst gap between an edge and the global fit -- a fit diagnostic.

        Per-trial anchoring does not use the fit for anchored trials, so this
        measures clock drift and edge jitter, not the alignment error.
        """
        return float(np.abs(self.residuals).max()) if self.residuals.size else np.nan

    def __repr__(self):
        return (f"<Alignment slope={self.slope:.9f} offset={self.intercept:.4f}s "
                f"{self.n_pulses} pulses / {self.n_trials} trials, "
                f"{self.n_anchored} anchored, "
                f"max fit resid {self.max_residual * 1e3:.2f} ms>")


def _nearest_within(sorted_vals, targets, tol):
    """Index into sorted_vals of the nearest value to each target, else -1."""
    targets = np.asarray(targets, dtype=float)
    if sorted_vals.size == 1:
        idx = np.zeros(targets.size, dtype=np.int64)
    else:
        # sorted_vals is sorted, so the nearest is one of the two straddling it.
        j = np.clip(np.searchsorted(sorted_vals, targets), 1, sorted_vals.size - 1)
        left, right = sorted_vals[j - 1], sorted_vals[j]
        idx = np.where(targets - left <= right - targets, j - 1, j)
    ok = np.abs(sorted_vals[idx] - targets) <= tol
    return np.where(ok, idx, -1)


def _modal_offset(ts, rises, window, max_pairs=4_000_000):
    """The (rise - trial_start) value shared by the most pairs.

    Correctly paired trials all sit at nearly the same offset; wrong pairings
    scatter. Taking the densest cluster therefore recovers the true offset
    without assuming which rise belongs to which trial -- which is the point,
    since a dropped or spurious pulse is exactly what breaks positional
    pairing. ``window`` must be wide enough to hold the whole session's clock
    drift, since the slope is not known yet.
    """
    step = max(1, int(np.ceil(ts.size * rises.size / max_pairs)))
    d = np.sort((rises[None, :] - ts[::step, None]).ravel())
    lo = np.searchsorted(d, d - window, side="left")
    hi = np.searchsorted(d, d + window, side="right")
    best = int(np.argmax(hi - lo))
    return float(np.median(d[lo[best]:hi[best]]))


def _assign(rises, predicted, tol):
    """Anchor index per trial, at most one trial per rise (nearest wins)."""
    idx = _nearest_within(rises, predicted, tol)
    # Two trials claiming one edge means the pairing is wrong, not merely
    # incomplete. Keep whichever trial sits closer; drop the other.
    order = np.argsort(np.where(idx >= 0, np.abs(rises[idx] - predicted), np.inf))
    taken = set()
    for i in order:
        if idx[i] < 0:
            continue
        if idx[i] in taken:
            idx[i] = -1
        else:
            taken.add(int(idx[i]))
    return idx


def align(session_data, photo: Photometry, fit_slope: bool = True,
          tol: float = 0.05) -> Alignment:
    """Map the Bpod clock onto the photometry clock using the sync line.

    Three passes, none of which assumes rise *i* belongs to trial *i*:

    1. **Robust offset.** The modal value of (rise - trial start) over all
       trial/rise combinations. Correct pairs share an offset; wrong pairs
       scatter, so the densest cluster wins.
    2. **Fit the rate.** Trials are matched to rises at that offset, and a line
       is fitted through the matched pairs. This supplies the clock *slope*,
       which a single edge cannot give.
    3. **Per-trial anchoring.** Each trial re-claims the rise nearest to where
       the fitted line puts its start, within ``tol`` seconds, at most one trial
       per rise. Those become ``Alignment.anchors``, and :class:`Session` hangs
       each trial's events off its own edge rather than off the line.

    Because pass 1 never assumes positional correspondence, a dropped or
    spurious pulse costs only the trials actually missing an edge -- the rest
    stay anchored, and the edgeless ones fall back to the fitted line. Compare
    ``n_anchored`` with ``n_trials`` to see whether any were lost.

    fit_slope=False forces slope 1 and fits only the offset.
    """
    ts = _times(session_data, "TrialStartTimestamp")
    rises = photo.sync_rises
    if min(ts.size, rises.size) < 2:
        raise ValueError(f"need >=2 sync pulses to align; got {rises.size} "
                         f"pulses and {ts.size} trials")

    # Pass 1. The window must absorb the session's whole clock drift, because
    # the slope is still unknown here -- 1e-5 over an hour is ~36 ms.
    coarse = max(tol, 0.25)
    idx = _assign(rises, ts + _modal_offset(ts, rises, coarse), coarse)

    # Pass 2. Fit only through trials that found an edge.
    m = idx >= 0
    if m.sum() < 2:
        raise ValueError(f"could not match trials to sync pulses: only "
                         f"{int(m.sum())} of {ts.size} trials found an edge")
    slope, intercept = (np.polyfit(ts[m], rises[idx[m]], 1) if fit_slope
                        else (1.0, float(np.mean(rises[idx[m]] - ts[m]))))

    # Pass 3. Re-assign against the fitted line, now at the tight tolerance.
    idx = _assign(rises, slope * ts + intercept, tol)
    m = idx >= 0
    anchors = np.where(m, rises[np.where(m, idx, 0)], np.nan)

    return Alignment(
        slope=float(slope),
        intercept=float(intercept),
        n_pulses=int(rises.size),
        n_trials=int(ts.size),
        residuals=rises[idx[m]] - (slope * ts[m] + intercept),
        anchors=anchors,
        trial_starts=ts.copy(),
    )


# --------------------------------------------------------------------------
# Session: behavior + photometry, aligned
# --------------------------------------------------------------------------

@dataclass
class Session:
    """A behavior session paired with its photometry recording."""

    behavior: object                  # raw SessionData, for lick_plotting
    photo: Photometry
    alignment: Alignment
    trials: pd.DataFrame
    name: str = ""
    animal: str = ""
    date: str = ""
    protocol: str = ""
    behavior_path: Path | None = None
    anchor: str = "trial"             # "trial" = own sync edge; "fit" = global line

    def _anchor(self, anchor):
        a = self.anchor if anchor is None else anchor
        if a not in ("trial", "fit"):
            raise ValueError(f"anchor must be 'trial' or 'fit', got {a!r}")
        return a

    def event_times(self, event, clock: str = "photometry",
                    anchor: str | None = None) -> np.ndarray:
        """Per-trial times of one event, one value per trial.

        ``event`` is a trial-table column of trial-relative times -- typically
        ``"<State>_on"`` or ``"<State>_off"`` (``"Cue1_on"``, ``"Reward_on"``,
        ...) -- or ``"trial_start"`` / ``"trial_end"``, already session clock.

        It may also be a tuple of such columns, for an event that lives in one
        of several states depending on the trial: ``("Cue1_on", "Cue2_on")`` is
        "whichever cue this trial played", ``("Reward_on", "NoReward_on")`` the
        outcome. Each trial takes the earliest of them it actually entered, NaN
        if none.

        clock: "photometry" (default), "bpod" (Bpod session clock), or "trial"
        (raw trial-relative, as stored).

        anchor: "trial" hangs each trial off its own sync rising edge (the
        default); "fit" puts everything through the global fitted line. Only
        affects clock="photometry". Defaults to ``self.anchor``.
        """
        df = self.trials
        start = df["trial_start"].to_numpy(dtype=float)
        if isinstance(event, str) and event in ("trial_start", "trial_end"):
            bpod = df[event].to_numpy(dtype=float)
            rel = bpod - start
        else:
            cols = (event,) if isinstance(event, str) else tuple(event)
            missing = [c for c in cols if c not in df.columns]
            if missing:
                raise KeyError(f"{missing} not in trial table; try one of "
                               f"{[c for c in df.columns if c.endswith(('_on', '_off'))]}")
            R = df[list(cols)].to_numpy(dtype=float)
            # fmin.reduce skips NaN where any column has a value, and leaves NaN
            # only where the trial entered none of the states.
            rel = np.fmin.reduce(R, axis=1)
            bpod = rel + start

        if clock == "trial":
            return rel
        if clock == "bpod":
            return bpod
        if clock == "photometry":
            if self._anchor(anchor) == "trial":
                return self.alignment.anchored(rel)
            return self.alignment.to_photometry(bpod)
        raise ValueError(f"unknown clock {clock!r}")

    def lick_times(self, clock: str = "photometry",
                   anchor: str | None = None) -> list[np.ndarray]:
        """Lick times per trial, as a list of arrays (ragged, so not a column)."""
        licks = [np.asarray(lk, dtype=float) for lk in self.trials["licks"]]
        if clock == "trial":
            return licks
        starts = self.trials["trial_start"].to_numpy(dtype=float)
        if clock == "bpod":
            return [lk + s for lk, s in zip(licks, starts)]
        if clock == "photometry":
            al = self.alignment
            if self._anchor(anchor) == "fit":
                return [al.to_photometry(lk + s) for lk, s in zip(licks, starts)]
            # Each trial's licks ride on that trial's own edge; trials without
            # one fall back to the global fit, as in Alignment.anchored.
            return [al.anchors[i] + lk * al.slope
                    if np.isfinite(al.anchors[i]) else al.to_photometry(lk + starts[i])
                    for i, lk in enumerate(licks)]
        raise ValueError(f"unknown clock {clock!r}")

    @property
    def cue_states(self) -> list[str]:
        """The protocol's cue states, ``["Cue1", "Cue2", ...]``, in name order."""
        return sorted(c[:-3] for c in self.trials.columns
                      if re.fullmatch(r"Cue\d+_on", c))

    def is_operant(self, tol: float = 0.01) -> bool:
        """True when the cue ends on the animal's response, not on a timer.

        In a classical protocol every cue state lasts the same fixed time, so
        the outcome sits a fixed delay after cue onset. In an operant one the
        cue ends at the first lick, so that delay varies trial to trial and
        nothing downstream of the cue can be derived from the sync line -- it
        has to come from Bpod's own state times. ``tol`` (seconds) absorbs
        Bpod's 0.1 ms state-timing jitter.
        """
        d = np.concatenate([
            (self.trials[f"{c}_off"] - self.trials[f"{c}_on"]).dropna().to_numpy()
            for c in self.cue_states] or [np.array([])])
        return bool(d.size > 1 and np.ptp(d) > tol)

    def signal(self, which: str = "F") -> np.ndarray:
        """A whole-session trace: "F", "iso", "F_subtr", or "F_iso_regressed"."""
        return signal_of(self.photo, which)

    def _events(self, event, trials=None, anchor=None) -> np.ndarray:
        """Photometry-clock event times, optionally subset by trial."""
        names = isinstance(event, str) or (
            isinstance(event, tuple) and all(isinstance(e, str) for e in event))
        ev = (self.event_times(event, "photometry", anchor=anchor) if names
              else np.atleast_1d(np.asarray(event, dtype=float)))
        return ev if trials is None else ev[np.asarray(trials)]

    def epoch(self, event, window=(-2.0, 5.0), which="F", trials=None,
              fill=np.nan, anchor=None):
        """Peri-event signal matrix.

        event   trial-table column name or tuple of them (see
                :meth:`event_times`), or an array of times already on the
                photometry clock.
        window  (before, after) in seconds; ``before`` is normally negative.
        which   which trace, per :meth:`signal`.
        trials  bool mask or index array, applied before extraction.
        fill    value for samples outside the recording.
        anchor  "trial" (own sync edge) or "fit"; defaults to ``self.anchor``.

        Returns (t, M): ``t`` is (n_samples,) seconds relative to the event,
        ``M`` is (n_events, n_samples). Rows whose event time is NaN (state not
        visited) are all ``fill``.
        """
        ev = self._events(event, trials, anchor)
        return epoch_signal(self.photo, ev, window, which, fill)

    def epoch_licks(self, event, window=(-2.0, 5.0), trials=None, anchor=None):
        """Lick times relative to an event, one array per trial (for rasters)."""
        ev = self._events(event, trials, anchor)
        licks = self.lick_times("photometry", anchor=anchor)
        if trials is not None:
            licks = [licks[i] for i in np.arange(len(licks))[np.asarray(trials)]]
        out = []
        for lk, e in zip(licks, ev):
            rel = lk - e if np.isfinite(e) else np.array([])
            out.append(rel[(rel >= window[0]) & (rel <= window[1])])
        return out

    def __repr__(self):
        return (f"<Session {self.name or self.animal} {self.date} "
                f"{len(self.trials)} trials, {self.photo.duration:.0f}s photometry, "
                f"align max resid {self.alignment.max_residual * 1e3:.2f} ms>")


def load_session_pair(behavior_path, photometry_path, fit_slope=True,
                      skip_pulses: int = 0) -> Session:
    """Load one behavior file and one photometry file and align them.

    skip_pulses  drop this many leading sync pulses before aligning, for a line
                 toggled before the first real trial.
    """
    behavior_path = Path(behavior_path)
    sd = load_session(behavior_path)
    if sd is None:
        raise ValueError(f"no SessionData in {behavior_path.name}")
    photo = load_photometry(photometry_path, skip_pulses=skip_pulses)
    meta = BEHAVIOR_RE.match(behavior_path.stem)
    return Session(
        behavior=sd,
        photo=photo,
        alignment=align(sd, photo, fit_slope=fit_slope),
        trials=trial_table(sd),
        name=behavior_path.stem,
        animal=meta["animal"] if meta else (photo.animal or ""),
        date=meta["date"] if meta else (photo.date or ""),
        protocol=meta["protocol"] if meta else "",
        behavior_path=behavior_path,
    )


# --------------------------------------------------------------------------
# Discovery / pairing
# --------------------------------------------------------------------------

def find_photometry_files(root, pattern="*Photodata.mat") -> list[Path]:
    root = Path(root)
    return [root] if root.is_file() else sorted(root.rglob(pattern))


def list_pairs(behavior_root, photometry_root, animals=None, dates=None):
    """What's available, as a DataFrame, without loading any of it.

    Use this to see what's in the folders before choosing what to hand to
    :func:`load_all` -- it only reads file names and sizes, so it stays fast as
    the folders grow. ``matched`` is False for a session missing its counterpart.
    """
    pairs, no_photo, no_beh = pair_files(behavior_root, photometry_root,
                                         animals=animals, dates=dates)
    rows = []
    for b, p in pairs:
        m = BEHAVIOR_RE.match(b.stem)
        rows.append({"animal": m["animal"] if m else "", "date": m["date"] if m else "",
                     "protocol": m["protocol"] if m else "", "matched": True,
                     "photometry MB": round(p.stat().st_size / 1e6, 1),
                     "behavior file": b.name, "photometry file": p.name})
    for b in no_photo:
        m = BEHAVIOR_RE.match(b.stem)
        rows.append({"animal": m["animal"] if m else "", "date": m["date"] if m else "",
                     "protocol": m["protocol"] if m else "", "matched": False,
                     "photometry MB": np.nan,
                     "behavior file": b.name, "photometry file": ""})
    for p in no_beh:
        m = PHOTO_RE.match(p.stem)
        rows.append({"animal": m["animal"] if m else "", "date": m["date"] if m else "",
                     "protocol": "", "matched": False,
                     "photometry MB": round(p.stat().st_size / 1e6, 1),
                     "behavior file": "", "photometry file": p.name})
    cols = ["animal", "date", "protocol", "matched", "photometry MB",
            "behavior file", "photometry file"]
    # An empty frame has no columns to sort on, so name them: a selection that
    # matched nothing should show an empty table, not raise a KeyError.
    return (pd.DataFrame(rows, columns=cols).sort_values(["animal", "date"],
                                                         kind="stable")
            .reset_index(drop=True))


def _as_set(v):
    """Normalise a filter argument to a set of strings, or None for 'all'.

    Accepts a bare string ("SW007"), any iterable of them, or None. Values are
    stringified so dates can be passed as ints (20260905) as well as strings.
    """
    if v is None:
        return None
    # A bare scalar -- "SW007", 20260905, np.str_ -- is a one-element filter,
    # not something to iterate (iterating a string would filter by letter).
    if isinstance(v, (str, bytes)) or not hasattr(v, "__iter__"):
        return {str(v)}
    return {str(x) for x in v}


def pair_files(behavior_root, photometry_root, animals=None, dates=None):
    """Match behavior files to photometry files on (animal, date).

    animals / dates restrict which sessions are considered -- a string, an
    iterable of strings, or None for everything. Filtering happens here, before
    anything is read, so a deselected session is never opened and never shows up
    in the unmatched lists::

        pair_files(B, P, animals="SW007")
        pair_files(B, P, animals=["SW007", "SW008"], dates="20260905")

    Returns (pairs, unmatched_behavior, unmatched_photometry), where pairs is a
    list of (behavior_path, photometry_path). Where one animal-day has several
    photometry runs, the one whose sync-pulse count matches nTrials wins; ties
    fall back to the lowest run number.
    """
    animals, dates = _as_set(animals), _as_set(dates)

    def selected(m):
        return ((animals is None or m["animal"] in animals)
                and (dates is None or m["date"] in dates))

    by_key: dict[tuple, list[Path]] = {}
    unmatched_photo = []
    for p in find_photometry_files(photometry_root):
        m = PHOTO_RE.match(p.stem)
        if m is None:
            unmatched_photo.append(p)   # name didn't parse -- worth flagging
        elif selected(m):
            by_key.setdefault((m["animal"], m["date"]), []).append(p)

    pairs, unmatched_beh, used = [], [], set()
    for b in find_session_files(behavior_root):
        m = BEHAVIOR_RE.match(b.stem)
        if m is not None and not selected(m):
            continue                    # deselected -- silently skipped
        candidates = [c for c in by_key.get((m["animal"], m["date"]), [])
                      if c not in used] if m else []
        if not candidates:
            unmatched_beh.append(b)
            continue

        chosen = candidates[0]
        if len(candidates) > 1:
            sd = load_session(b)
            want = n_trials(sd) if sd is not None else None
            chosen = next((c for c in candidates
                           if load_photometry(c).sync_rises.size == want), chosen)
        used.add(chosen)
        pairs.append((b, chosen))

    unmatched_photo += [p for ps in by_key.values() for p in ps if p not in used]
    return pairs, unmatched_beh, sorted(unmatched_photo)


def _short_pulse_hint(photo: Photometry, frac: float = 0.2) -> str:
    """One line naming the pulses too short to be a trial's ITI, if any.

    A count mismatch is nearly always the line being toggled outside a trial,
    which leaves a HIGH far shorter than an ITI. Saying *which* pulse and how
    short turns "101 pulses vs 100 trials" into something actionable.

    ``frac`` is deliberately well under the shortest real ITI (~4 s against a
    ~9 s median here): a stray toggle is an order of magnitude short, so a loose
    threshold would just list the short end of the ITI distribution.
    """
    w = photo.pulse_widths
    if w.size < 2:
        return "too few pulses to judge"
    med = float(np.nanmedian(w))
    odd = np.flatnonzero(w < frac * med)
    if odd.size == 0:
        return (f"all pulses are ITI-length (median {med:.2f}s) -- the extra edge "
                f"is not an obvious stray")
    where = ", ".join(f"#{i} ({w[i]:.2f}s)" for i in odd[:4])
    # Only a run at the very start is fixable by dropping leading pulses; a
    # stray in the middle needs looking at, not a count.
    fix = (f" -- pass skip_pulses={odd.size} to drop them"
           if np.array_equal(odd, np.arange(odd.size)) else "")
    return (f"pulses under {frac:.0%} of the {med:.2f}s median ITI: {where}{fix}"
            + (f" (+{odd.size - 4} more)" if odd.size > 4 else ""))


def skip_for(skip_pulses, animal: str, date: str) -> int:
    """How many leading pulses to drop for one session.

    ``skip_pulses`` is either a count for everything, or a dict keyed by
    ``"<animal>_<date>"`` (most specific), ``"<animal>"``, or ``"<date>"`` --
    a mistake usually belongs to one session, occasionally to a whole day::

        skip_pulses=1                            # every session loaded
        skip_pulses={"SW008_20260906": 1}        # just that one
        skip_pulses={"SW008_20260906": 1, "SW010_20260906": 1}
    """
    if skip_pulses is None:
        return 0
    if not isinstance(skip_pulses, dict):
        return int(skip_pulses)
    for key in (f"{animal}_{date}", animal, date):
        if key in skip_pulses:
            return int(skip_pulses[key])
    return 0


def load_all(behavior_root, photometry_root, animals=None, dates=None,
             verbose=True, skip_pulses=None) -> dict[str, Session]:
    """Load and align every selected behavior/photometry pair under the roots.

    animals / dates pick which sessions to load -- a string, an iterable of
    them, or None for everything (see :func:`pair_files`). Deselected sessions
    are never opened, which matters because each photometry file is ~45 MB::

        load_all(B, P)                              # everything
        load_all(B, P, animals="SW007")             # one animal, all its days
        load_all(B, P, dates="20260905")            # one day, all animals
        load_all(B, P, animals=["SW007", "SW010"], dates="20260905")

    skip_pulses drops leading sync pulses that are not trials, per :func:`skip_for`.
    A session that needs it announces itself: the load line flags any pulse/trial
    mismatch, and names the short pulse that is the usual cause.

    Returns {"<animal>_<date>": Session}. Selected files with no counterpart are
    reported and skipped.
    """
    pairs, no_photo, no_beh = pair_files(behavior_root, photometry_root,
                                         animals=animals, dates=dates)
    sessions = {}
    for b, p in pairs:
        m = BEHAVIOR_RE.match(b.stem)
        skip = skip_for(skip_pulses, m["animal"], m["date"]) if m else 0
        sess = load_session_pair(b, p, skip_pulses=skip)
        sessions[f"{sess.animal}_{sess.date}"] = sess
        if verbose:
            a = sess.alignment
            warn = "" if a.n_pulses == a.n_trials else \
                f"  !! {a.n_pulses} pulses vs {a.n_trials} trials"
            if a.n_anchored < a.n_trials:
                warn += (f"  !! {a.n_trials - a.n_anchored} trials without their "
                         f"own edge (using the global fit)")
            if skip:
                warn += f"  (first {skip} pulse(s) dropped)"
            print(f"{sess.animal} {sess.date}  {len(sess.trials):>4d} trials  "
                  f"{sess.photo.duration:7.1f}s  "
                  f"{a.n_anchored} anchored  "
                  f"fit resid {a.max_residual * 1e3:5.2f} ms{warn}")
            if a.n_pulses != a.n_trials:
                print("   " + _short_pulse_hint(sess.photo))
    if verbose:
        for b in no_photo:
            print(f"no photometry for  {b.name}")
        for p in no_beh:
            print(f"no behavior for    {p.name}")
    return sessions


# --------------------------------------------------------------------------
# Signal normalisation
# --------------------------------------------------------------------------

def baseline_subtract(M, t, window=(-2.0, 0.0)):
    """Subtract each row's mean over a pre-event window.

    ``F`` in these files is already normalised (values near 0, not raw counts),
    so subtraction -- not division -- is the right baseline here.
    """
    m = (t >= window[0]) & (t <= window[1])
    if not m.any():
        raise ValueError(f"baseline window {window} has no samples in t")
    M = np.asarray(M, dtype=float)
    return M - np.nanmean(M[:, m], axis=1, keepdims=True)


def zscore_baseline(M, t, window=(-2.0, 0.0)):
    """Z-score each row against its own pre-event window."""
    m = (t >= window[0]) & (t <= window[1])
    M = np.asarray(M, dtype=float)
    sd = np.nanstd(M[:, m], axis=1, keepdims=True)
    return (M - np.nanmean(M[:, m], axis=1, keepdims=True)) / np.where(sd == 0, np.nan, sd)


def mean_sem(M):
    """(mean, sem) across rows, ignoring NaNs."""
    M = np.asarray(M, dtype=float)
    n = np.sum(np.isfinite(M), axis=0)
    return np.nanmean(M, axis=0), np.nanstd(M, axis=0) / np.sqrt(np.maximum(n, 1))


if __name__ == "__main__":
    sessions = load_all(BEHAVIOR_ROOT, PHOTOMETRY_ROOT)
    for name, s in sessions.items():
        print(f"\n{name}: {s}")
        print(s.trials.drop(columns=["licks"]).head())
