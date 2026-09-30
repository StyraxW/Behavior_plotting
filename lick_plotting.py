"""Python port of the MATLAB behavior_plotting.m lick-count plotter.

Counts the licks in a fixed window after each Bpod trial's cue onset
(LICK_WINDOW, 1.5 s by default), measures the first-lick latency in the same
window, and infers the block structure, ready to be plotted against trial
number.

Point load_counts() at an animal folder (or any parent folder) and it finds
every "Session Data" folder underneath, loads each session, counts its licks
and infers its block structure. The drawing itself lives in
lick_plotting.ipynb, so colors, markers, guide lines and limits can be changed
in the notebook without touching this module.
"""

from fnmatch import fnmatch
from pathlib import Path

import numpy as np
import scipy.io as sio

# Lick event field names, in order of preference. The upstairs rig records
# licks on BNC1; the NPX rig records them on Port1.
LICK_EVENTS = ("BNC1High", "Port1In")

# Seconds after cue onset to count licks in. A fixed window keeps the counts
# comparable across trials and protocols, where the Cue state's own duration
# does not. None counts over the Cue state instead (the MATLAB behavior).
LICK_WINDOW = 1.5

# Bpod writes sessions into a folder with this name, alongside a
# "Session Settings" folder whose DefaultSettings.mat is NOT a session.
SESSION_DIR_NAME = "session data"

# Background shading per session type: (trial_start, trial_stop, color).
# Mirrors the patch() calls in the MATLAB version.
BLOCK_SHADING = {
    "1": [(0, 200, "r")],
    "2": [(0, 200, "b")],
    "3A": [(0, 100, "r"), (100, 200, "b")],
    "3B": [(0, 100, "b"), (100, 200, "r")],
    "3AU": [(0, 70, "r"), (70, 140, "b"), (140, 210, "r")],
    "3BU": [(0, 70, "b"), (70, 140, "r"), (140, 210, "b")],
    "rec": [(0, 60, "r"), (60, 120, "b"), (120, 180, "r"), (180, 240, "b")],
}

# Block colors, keyed by which cue is rewarded in the block.
CUE_COLORS = {1: "r", 2: "b"}

TRANSPARENCY = 0.1


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def list_stages(root):
    """Names of the task-stage folders directly under an animal folder.

    These are the names accepted by the ``stages`` argument of
    find_session_files() / load_tree() / load_counts().
    """
    root = Path(root)
    return sorted(
        folder.name for folder in root.iterdir()
        if folder.is_dir() and folder.name.lower() != SESSION_DIR_NAME
    )


def _as_patterns(stages):
    """Normalize the stages argument to a list of patterns, or None."""
    if stages is None:
        return None
    if isinstance(stages, (str, Path)):
        stages = [stages]
    return [str(s).strip().strip("/\\") for s in stages]


def _in_stages(path, root, patterns):
    """True if any folder between root and path matches a stage pattern.

    Matching is case-insensitive and accepts glob wildcards, so
    stages="Inference*" picks up every inference stage.
    """
    try:
        parts = path.relative_to(root).parts[:-1]  # drop the file name
    except ValueError:
        return False
    return any(
        fnmatch(part.lower(), pat.lower())
        for part in parts for pat in patterns
    )


def find_session_files(root, pattern="*.mat", stages=None, verbose=False):
    """Find every Bpod session file under root, recursively.

    Only looks inside folders named "Session Data", so sibling
    "Session Settings\\DefaultSettings.mat" files are skipped. Returns paths
    sorted by name, which puts the yyyymmdd_HHMMSS suffixes in date order.

    stages restricts the search to selected task-stage folders under root:
    one folder name, or a list of them, matched case-insensitively against
    every folder on the path and accepting glob wildcards. None (the default)
    takes every session under root -- the previous behavior. Use
    list_stages(root) to see what a given animal folder offers.
    """
    root = Path(root)
    if root.is_file():
        return [root]

    # If root is itself a Session Data folder, take it directly.
    if root.name.lower() == SESSION_DIR_NAME:
        return sorted(root.glob(pattern))

    files = []
    for folder in root.rglob("*"):
        if folder.is_dir() and folder.name.lower() == SESSION_DIR_NAME:
            files.extend(sorted(folder.glob(pattern)))

    patterns = _as_patterns(stages)
    if patterns is not None:
        files = [f for f in files if _in_stages(f, root, patterns)]
        if verbose:
            for pat in patterns:
                if not any(_in_stages(f, root, [pat]) for f in files):
                    print(f"no sessions matched stage {pat!r}; "
                          f"available: {list_stages(root)}")

    return sorted(files, key=lambda p: (p.parent.parent.name, p.name))


def load_session(path):
    """Load one Bpod .mat file and return its SessionData struct.

    Returns None if the file holds no SessionData (e.g. a settings file).
    """
    mat = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    return mat.get("SessionData")


def load_folder(folder, pattern="*.mat"):
    """Load every .mat file directly inside one folder.

    Returns {file stem: SessionData}, sorted by file name.
    """
    sessions = {}
    for path in sorted(Path(folder).glob(pattern)):
        session_data = load_session(path)
        if session_data is not None:
            sessions[path.stem] = session_data
    return sessions


def load_tree(root, pattern="*.mat", stages=None):
    """Load every session under root, recursively.

    Returns {relative label: SessionData}, where the label is
    "<protocol folder>/<file stem>" so sessions from different protocols
    stay distinguishable. stages selects which task-stage folders under root
    to load; see find_session_files().
    """
    root = Path(root)
    sessions = {}
    for path in find_session_files(root, pattern=pattern, stages=stages):
        session_data = load_session(path)
        if session_data is None:
            continue
        sessions[session_label(path, root)] = session_data
    return sessions


def session_label(path, root=None):
    """Readable label for a session file: "<protocol>/<file stem>"."""
    path = Path(path)
    protocol = path.parent.parent.name  # parent of the "Session Data" folder
    return f"{protocol}/{path.stem}" if protocol else path.stem


# --------------------------------------------------------------------------
# Struct helpers
# --------------------------------------------------------------------------

def _fields(struct):
    return set(getattr(struct, "_fieldnames", []))


def _times(struct, name):
    """Read a state/event field as a 1-D float array (empty if absent)."""
    if name not in _fields(struct):
        return np.array([], dtype=float)
    return np.atleast_1d(np.asarray(getattr(struct, name), dtype=float))


def _entered(struct, name):
    """True if a Bpod state was visited (its timestamps are not NaN)."""
    t = _times(struct, name)
    return bool(t.size) and not np.isnan(t[0])


def find_lick_event(session_data):
    """Pick the lick event field actually present in this session."""
    trials = np.atleast_1d(session_data.RawEvents.Trial)
    present = set()
    for trial in trials:
        present |= _fields(trial.Events)
    for name in LICK_EVENTS:
        if name in present:
            return name
    raise ValueError(f"no lick event found; trials have {sorted(present)}")


def n_trials(session_data):
    return int(np.atleast_1d(session_data.nTrials).ravel()[0])


# --------------------------------------------------------------------------
# Lick counting and latency
# --------------------------------------------------------------------------

def cue_per_trial(session_data):
    """Which cue each trial presented, as a 1-D array of 1 or 2."""
    trials = np.atleast_1d(session_data.RawEvents.Trial)
    return np.array(
        [1 if _entered(trials[i].States, "Cue1") else 2
         for i in range(n_trials(session_data))]
    )


def _cue_licks(session_data, lick_event=None, window=LICK_WINDOW):
    """Per trial, the licks inside the cue window relative to cue onset.

    Yields (which_cue, trial_number, licks) where licks is a sorted array of
    lick times in seconds after cue onset, so len(licks) is the lick count and
    licks[0] is the first-lick latency. Shared by count_licks() and
    first_lick_latency() so both read the same window the same way.
    """
    if lick_event is None:
        lick_event = find_lick_event(session_data)

    trials = np.atleast_1d(session_data.RawEvents.Trial)

    for i in range(n_trials(session_data)):
        trial = trials[i]
        cue_1 = _times(trial.States, "Cue1")
        if cue_1.size and not np.isnan(cue_1[0]):
            cue, which = cue_1, 1
        else:
            cue, which = _times(trial.States, "Cue2"), 2

        licks = _times(trial.Events, lick_event)
        if licks.size and cue.size and not np.isnan(cue[0]):
            start = cue[0]
            stop = start + window if window is not None else cue[1]
            inside = np.sort(licks[(licks >= start) & (licks <= stop)]) - start
        else:
            inside = np.empty(0)

        yield which, i + 1, inside  # 1-based trial number, as in MATLAB


def _by_cue(per_trial):
    """Split [(which, trial, value)] into the two (2, n) arrays we plot."""
    rows = {1: [], 2: []}
    for which, trial, value in per_trial:
        rows[which].append((value, trial))
    return (
        np.array(rows[1], dtype=float).T.reshape(2, -1),
        np.array(rows[2], dtype=float).T.reshape(2, -1),
    )


def count_licks(session_data, lick_event=None, window=LICK_WINDOW):
    """Count licks in a fixed window after each trial's cue onset.

    window is the window length in seconds, measured from cue onset, so the
    count does not depend on how long the cue state itself happened to run.
    Pass window=None to fall back to the cue state's own start/end times (the
    MATLAB behavior).

    Returns (count_1, count_2): two (2, n) arrays whose first row is the lick
    count and second row the 1-based trial number, matching the MATLAB layout.
    Cue1 trials go in count_1, Cue2 trials in count_2.
    """
    return _by_cue(
        (which, trial, len(licks))
        for which, trial, licks in _cue_licks(session_data, lick_event, window)
    )


def first_lick_latency(session_data, lick_event=None, window=LICK_WINDOW,
                       miss=None):
    """Latency of the first lick after cue onset, in seconds.

    Trials with no lick inside the window get `miss`, which defaults to the
    window length (1.5 s with the default window) -- a miss is "at least the
    whole window", so it plots at the ceiling rather than dropping out. Pass
    miss=np.nan to leave misses out of the trace instead.

    Returns (latency_1, latency_2) shaped exactly like count_licks(): two
    (2, n) arrays, first row the latency in seconds, second row the 1-based
    trial number. Cue1 trials go in latency_1, Cue2 trials in latency_2.
    """
    if miss is None:
        miss = window if window is not None else np.nan

    return _by_cue(
        (which, trial, licks[0] if licks.size else miss)
        for which, trial, licks in _cue_licks(session_data, lick_event, window)
    )


# --------------------------------------------------------------------------
# Block-structure inference
# --------------------------------------------------------------------------

def rewarded_per_trial(session_data):
    """True where the trial ended in the Reward state."""
    trials = np.atleast_1d(session_data.RawEvents.Trial)
    return np.array(
        [_entered(trials[i].States, "Reward")
         for i in range(n_trials(session_data))]
    )


def block_edges(session_data):
    """Trial indices bounding each block, as [0, e1, e2, ..., nTrials].

    Uses SessionData.BlockTransition when it marks any transition; otherwise
    falls back to detecting where the rewarded cue changes.
    """
    total = n_trials(session_data)

    if "BlockTransition" in _fields(session_data):
        flags = np.atleast_1d(np.asarray(session_data.BlockTransition)).ravel()
        marks = [int(i) + 1 for i in np.flatnonzero(flags[:total])]
        if marks:
            edges = [0] + marks
            if edges[-1] < total:
                edges.append(total)
            return edges

    # Fallback: block identity is whichever cue pays off, so a change in the
    # rewarded cue marks a boundary.
    cues = cue_per_trial(session_data)
    rewarded = rewarded_per_trial(session_data)
    idx = np.flatnonzero(rewarded)
    if idx.size == 0:
        return [0, total]

    labels = cues[idx]
    edges = [0]
    for k in range(1, labels.size):
        if labels[k] != labels[k - 1]:
            # Boundary sits between the two rewarded trials that disagree.
            edges.append(int((idx[k - 1] + idx[k]) // 2) + 1)
    edges.append(total)
    return edges


def infer_blocks(session_data):
    """Infer shading blocks from the data: (start, stop, color) per block.

    The block's color reports which cue was rewarded in it -- red for Cue1,
    blue for Cue2 -- so the shading is read off the file rather than assumed
    from a session-type string. Blocks with no clear winner are left unshaded.
    """
    cues = cue_per_trial(session_data)
    rewarded = rewarded_per_trial(session_data)
    edges = block_edges(session_data)

    blocks = []
    for start, stop in zip(edges[:-1], edges[1:]):
        if stop <= start:
            continue
        block_cues = cues[start:stop]
        block_rewarded = rewarded[start:stop]

        # Reward rate for each cue within this block.
        rates = {}
        for cue in (1, 2):
            mask = block_cues == cue
            rates[cue] = block_rewarded[mask].mean() if mask.any() else np.nan

        winner = None
        if not np.isnan(rates[1]) and not np.isnan(rates[2]):
            if rates[1] > rates[2]:
                winner = 1
            elif rates[2] > rates[1]:
                winner = 2
        elif not np.isnan(rates[1]) and rates[1] > 0:
            winner = 1
        elif not np.isnan(rates[2]) and rates[2] > 0:
            winner = 2

        if winner is not None:
            blocks.append((start, stop, CUE_COLORS[winner]))
    return blocks


def blocks_from_transitions(session_data, first_color="r"):
    """Blocks from BlockTransition with colors simply alternating.

    Kept for the case where reward contingency is not what alternates.
    """
    edges = block_edges(session_data)
    colors = ("r", "b") if first_color == "r" else ("b", "r")
    return [
        (edges[k], edges[k + 1], colors[k % 2]) for k in range(len(edges) - 1)
    ]


def describe_blocks(session_data):
    """One-line summary of the inferred block structure, for printing."""
    parts = [
        f"{start + 1}-{stop} Cue{1 if color == 'r' else 2}"
        for start, stop, color in infer_blocks(session_data)
    ]
    return " | ".join(parts) if parts else "no blocks detected"


# --------------------------------------------------------------------------
# Session bundles (data only -- the drawing lives in lick_plotting.ipynb)
# --------------------------------------------------------------------------

def session_counts(session_data, type="auto", lick_event=None,
                   window=LICK_WINDOW, miss=None):
    """Everything a plot needs from one session, with no matplotlib involved.

    Returns a dict with:
        count_1, count_2      (2, n) arrays from count_licks()
        latency_1, latency_2  (2, n) arrays from first_lick_latency()
        blocks                [(start, stop, color)] background shading
        n_trials              trial count
        lick_event            which event field the licks came from
        window                the counting window used, in seconds
        data                  the raw SessionData, for anything else

    Both metrics come from the same window, so the notebook can switch
    between plotting counts and latencies without reloading anything.

    type selects the shading: "auto" infers it from the file (default), any
    key of BLOCK_SHADING uses those fixed trial ranges, and "" disables it.
    window is the lick-counting window after cue onset; see count_licks().
    miss is the latency given to trials with no lick; see
    first_lick_latency().
    """
    if lick_event is None:
        lick_event = find_lick_event(session_data)

    count_1, count_2 = count_licks(session_data, lick_event=lick_event,
                                   window=window)
    latency_1, latency_2 = first_lick_latency(session_data,
                                              lick_event=lick_event,
                                              window=window, miss=miss)

    if type == "auto":
        blocks = infer_blocks(session_data)
    else:
        blocks = BLOCK_SHADING.get(str(type), [])

    return {
        "count_1": count_1,
        "count_2": count_2,
        "latency_1": latency_1,
        "latency_2": latency_2,
        "blocks": blocks,
        "n_trials": n_trials(session_data),
        "lick_event": lick_event,
        "window": window,
        "data": session_data,
    }


def load_counts(root, type="auto", pattern="*.mat", stages=None, verbose=True,
                window=LICK_WINDOW, miss=None):
    """Load and count every session under root: {label: session_counts(...)}.

    This is the slow step -- it reads the .mat files -- so the notebook runs
    it once and re-plots from the returned dict as often as it likes.

    stages limits the load to selected task-stage folders under root -- one
    name or a list, case-insensitive, glob wildcards allowed. For example
    load_counts(SW010, stages="Inference_Phase1") skips the
    WaterDeliveryPhotometry_DA_stim sessions. None loads everything.

    window is the lick-counting window after cue onset; see count_licks();
    miss is passed through to first_lick_latency().
    """
    root = Path(root)
    sessions = {}
    files = find_session_files(root, pattern=pattern, stages=stages,
                               verbose=verbose)
    for path in files:
        session_data = load_session(path)
        if session_data is None:
            if verbose:
                print(f"skipped (no SessionData): {path.name}")
            continue

        label = session_label(path, root)
        sessions[label] = session_counts(session_data, type=type,
                                         window=window, miss=miss)
        if verbose:
            span = f"{window}s after cue" if window is not None else "cue state"
            print(f"{label}  |  {sessions[label]['n_trials']} trials  |  "
                  f"licks on {sessions[label]['lick_event']} in {span}  |  "
                  f"blocks: {describe_blocks(session_data)}")
    return sessions
