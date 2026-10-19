"""Diarization: who spoke when. VAD regions -> voiceprints per window -> clusters -> turns.

Two clustering methods, picked with [diar] method in config.toml:
    "ahc"       agglomerative: keep merging the two closest groups until they are further
                apart than threshold. One knob, easy to reason about.
    "spectral"  NME-SC (Park et al. 2019): builds a "who sounds like whom" graph and reads the
                speaker count from its eigenvalues. No threshold to tune.
"""
from dataclasses import asdict, dataclass

import kaldi_native_fbank as knf
import numpy as np
import onnxruntime as ort
import soundfile as sf
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.cluster.vq import ClusterError, kmeans2
from scipy.linalg import eigh

from pipeline.config import SAMPLE_RATE, DiarConfig
from pipeline.vad import SpeechRegion


# Fixed design choices, not tuning knobs (so not in config.toml)
WINDOW_MS = 1500        # one voiceprint per 1.5 s window...
STEP_MS = 750           # ...every 0.75 s (50% overlap)
MIN_WINDOW_MS = 500     # shorter regions get no voiceprint; merge gives them the nearest speaker
FRAME_MS = 10           # fbank frame shift
N_MELS = 80
BATCH = 32
THREADS = 1             # core 0; whisper-server has cores 1-3
MERGE_GAP_MS = 500      # same speaker, gap <= this -> one turn
MIN_TURN_MS = 500       # shorter turns glued to a neighbour take its speaker

# Spectral clustering
MAX_SPEAKERS = 8        # the eigengap search looks for at most this many speakers
SC_MIN_WINDOWS = 20     # fewer windows (~15 s of speech) are too few for a stable graph: use AHC
SC_MAX_WINDOWS = 1000   # eigh is O(n^3): beyond this, cluster an even subsample and give
                        # every window the speaker of the nearest centroid
SC_P_STEPS = 20         # how many neighbour counts p to try...
SC_MAX_P_RATIO = 0.25   # ...from 2 up to 25% of the windows
KMEANS_RUNS = 10        # k-means++ restarts; the tightest result wins

SAMPLES_PER_MS = SAMPLE_RATE // 1000


@dataclass(frozen=True)
class Tuning:
    """The fixed design choices above as one object, so eval/bench_diar.py can vary them.
    The pipeline always uses the defaults."""
    window_ms: int = WINDOW_MS
    step_ms: int = STEP_MS
    cmn: str = "window"         # mean removed per "window", per "region", or "none"
    linkage: str = "average"    # AHC: how the distance between two groups is measured
    sc_max_p_ratio: float = SC_MAX_P_RATIO
    merge_gap_ms: int = MERGE_GAP_MS
    min_turn_ms: int = MIN_TURN_MS

    def __post_init__(self):
        if self.cmn not in ("window", "region", "none"):
            raise ValueError(f"cmn must be 'window', 'region' or 'none', got {self.cmn!r}")


@dataclass(frozen=True)
class Turn:
    start_ms: int
    end_ms: int
    speaker: str

    @property
    def duration_ms(self):
        return self.end_ms - self.start_ms

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class Window:
    start_ms: int
    end_ms: int
    region: int     # index of the VAD region this window belongs to

    @property
    def center_ms(self):
        return (self.start_ms + self.end_ms) // 2



def diarize(wav, regions, config, tuning=Tuning()):
    """Returns (turns, report). The report is printed by run.py to help tune the clustering."""
    embedder = SpeakerEmbedder(config.model)
    windows, embs = embed_windows(wav, regions, embedder, tuning)
    return cluster_turns(windows, embs, regions, config, tuning)


def cluster_turns(windows, embs, regions, config, tuning=Tuning()):
    """Voiceprints -> speakers -> turns. Separate from diarize() so the benchmark can try many
    clusterings on voiceprints it computed once."""
    if not windows:
        return [], {"method": config.method, "windows": 0, "speakers": 0, "centroid_similarity": []}

    x = l2norm(embs)
    if config.method == "spectral" and len(x) >= SC_MIN_WINDOWS:
        method = "spectral"
        labels = spectral(x, config.num_speakers, tuning.sc_max_p_ratio)
    else:
        method = "ahc"
        labels = ahc(x, config.threshold, config.num_speakers, tuning.linkage)

    if config.num_speakers == 0:
        labels = absorb_small(x, labels, max(1, config.min_cluster_ms // tuning.step_ms))
    labels = by_first_appearance(labels)

    report = {
        "method": method,
        "windows": len(windows),
        "speakers": int(labels.max()) + 1,
        "centroid_similarity": centroid_similarity(x, labels).round(2).tolist(),
    }
    turns = label_segments(windows, labels, regions)
    turns = merge_adjacent(turns, tuning.merge_gap_ms)
    return smooth(turns, tuning.min_turn_ms, tuning.merge_gap_ms), report



# Voiceprints

class SpeakerEmbedder:

    def __init__(self, model_path):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = THREADS
        opts.inter_op_num_threads = 1

        self.session = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name
        meta = self.session.get_modelmeta().custom_metadata_map

        # sherpa-onnx exports record how the model was trained:
        # normalize_samples "0" = it saw samples in int16 range, not [-1, 1]
        self.sample_scale = 32768.0 if meta.get("normalize_samples", "1") == "0" else 1.0

    def __call__(self, feats):
        """(batch, frames, 80) -> (batch, dim). All windows in one batch have the same length."""
        return self.session.run(None, {self.input: feats})[0]


def fbank(samples, scale):
    """float32 samples in [-1, 1] -> (n_frames, 80) log-mel features, one frame per 10 ms."""
    opts = knf.FbankOptions()
    opts.frame_opts.samp_freq = SAMPLE_RATE
    opts.frame_opts.dither = 0.0            # no random noise: same audio -> same features
    opts.frame_opts.snip_edges = True
    opts.mel_opts.num_bins = N_MELS

    fb = knf.OnlineFbank(opts)
    fb.accept_waveform(SAMPLE_RATE, (samples * scale).tolist())
    fb.input_finished()

    if fb.num_frames_ready == 0:
        return np.zeros((0, N_MELS), dtype=np.float32)
    return np.stack([fb.get_frame(i) for i in range(fb.num_frames_ready)]).astype(np.float32)


def frame_windows(n_frames, window_ms=WINDOW_MS, step_ms=STEP_MS):
    """(start, end) in frame indices. The last window ends exactly at the region end."""
    win = window_ms // FRAME_MS
    step = step_ms // FRAME_MS
    min_len = MIN_WINDOW_MS // FRAME_MS

    if n_frames < min_len:
        return []
    if n_frames <= win:
        return [(0, n_frames)]              # short region: one window = the whole region

    starts = list(range(0, n_frames - win + 1, step))
    if starts[-1] + win < n_frames:
        starts.append(n_frames - win)       # don't lose the tail of the region
    return [(s, s + win) for s in starts]


def embed_windows(wav, regions, embedder, tuning=Tuning()):
    windows = []
    embs = []
    pending = {}    # window length -> batch of (Window, features)

    def flush(length):
        batch = pending.pop(length)
        out = embedder(np.stack([f for _, f in batch]))
        windows.extend(w for w, _ in batch)
        embs.extend(out)

    for r_idx, region in enumerate(regions):
        samples, _ = sf.read(wav, start=region.start_ms * SAMPLES_PER_MS, stop=region.end_ms * SAMPLES_PER_MS, dtype="float32")
        feats = fbank(samples, embedder.sample_scale)     # once per region, not per window
        if tuning.cmn == "region" and len(feats):
            feats = feats - feats.mean(axis=0)

        for f0, f1 in frame_windows(len(feats), tuning.window_ms, tuning.step_ms):
            w_feats = feats[f0:f1]
            if tuning.cmn == "window":
                w_feats = w_feats - w_feats.mean(axis=0)     # CMN: removes mic colouring
            w = Window(region.start_ms + f0 * FRAME_MS, region.start_ms + f1 * FRAME_MS, r_idx)
            pending.setdefault(f1 - f0, []).append((w, w_feats))
            if len(pending[f1 - f0]) == BATCH:
                flush(f1 - f0)

    for length in list(pending):
        flush(length)

    if not windows:
        return [], np.zeros((0, 0), dtype=np.float32)

    # batches of different lengths finish out of order: sort back into time order
    order = sorted(range(len(windows)), key=lambda i: windows[i].start_ms)
    return [windows[i] for i in order], np.stack([embs[i] for i in order])



# Clustering

def l2norm(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)


def ahc(x, threshold, num_speakers, method="average"):
    """Agglomerative clustering: keep merging the two closest groups (average cosine distance)
    until the closest pair is further apart than threshold (or num_speakers groups are left).
    method "complete": the distance of two groups is their furthest pair instead.
    "ward": euclidean (x is L2-normalised), so threshold is on another scale than cosine."""
    if len(x) < 2:
        return np.zeros(len(x), dtype=int)

    metric = "euclidean" if method in ("ward", "centroid", "median") else "cosine"
    tree = linkage(x, method=method, metric=metric)
    if num_speakers > 0:
        labels = fcluster(tree, t=num_speakers, criterion="maxclust")
    else:
        labels = fcluster(tree, t=threshold, criterion="distance")
    return labels - 1   # fcluster counts from 1


def spectral(x, num_speakers, max_p_ratio=SC_MAX_P_RATIO):
    """NME-SC on at most SC_MAX_WINDOWS windows; the rest go to the nearest speaker centroid."""
    if len(x) <= SC_MAX_WINDOWS:
        return nme_sc(x, num_speakers, max_p_ratio)

    pick = np.linspace(0, len(x) - 1, SC_MAX_WINDOWS).round().astype(int)   # evenly over time
    sub = x[pick]
    sub_labels = nme_sc(sub, num_speakers, max_p_ratio)

    ids = np.unique(sub_labels)
    centroids = l2norm(np.stack([sub[sub_labels == k].mean(axis=0) for k in ids]))
    return ids[(x @ centroids.T).argmax(axis=1)]


def nme_sc(x, num_speakers, max_p_ratio=SC_MAX_P_RATIO):
    """Spectral clustering with the normalized maximum eigengap (Park et al. 2019).

    For each p: keep only every window's p most similar windows -> graph -> Laplacian.
    k speakers = k groups that barely connect = k eigenvalues near 0, then a jump (the eigengap).
    The best p has a large, clear jump while keeping the graph sparse: it minimises p / gap.
    """
    sim = (x @ x.T).astype(np.float64)     # cosine similarity (x is L2-normalised)
    max_k = min(MAX_SPEAKERS, len(x) - 1)
    best = None

    for p in neighbour_counts(len(x), max_p_ratio):
        lap = laplacian(binarize(sim, p))
        lam = eigh(lap, eigvals_only=True)          # ascending
        gaps = np.diff(lam[:max_k + 1])
        gap = gaps.max() / (lam[-1] + 1e-10)        # normalised: comparable across p
        ratio = p / max(gap, 1e-10)

        if best is None or ratio < best[0]:
            best = (ratio, lap, int(gaps.argmax()) + 1)

    _, lap, k = best
    if num_speakers > 0:
        k = num_speakers
    if k == 1:
        return np.zeros(len(x), dtype=int)

    # the k smallest eigenvectors place every window near its speaker's corner: k-means finds them
    _, vecs = eigh(lap, subset_by_index=[0, k - 1])
    return kmeans(vecs, k)


def neighbour_counts(n, max_p_ratio=SC_MAX_P_RATIO):
    top = max(2, int(n * max_p_ratio))
    return np.unique(np.linspace(2, top, SC_P_STEPS).astype(int))


def binarize(sim, p):
    """Each window keeps its p most similar windows (itself included) as 1, the rest 0."""
    top = np.argpartition(-sim, p - 1, axis=1)[:, :p]
    a = np.zeros_like(sim)
    np.put_along_axis(a, top, 1.0, axis=1)
    return 0.5 * (a + a.T)      # symmetric: "A is near B" counts half if B doesn't agree


def laplacian(a):
    a = a.copy()
    np.fill_diagonal(a, 0.0)
    return np.diag(a.sum(axis=1)) - a


def kmeans(v, k):
    """Best of several k-means++ runs: the one with the smallest within-cluster distance."""
    best = None
    for seed in range(KMEANS_RUNS):
        try:
            centroids, labels = kmeans2(v, k, minit="++", seed=seed, missing="raise")
        except ClusterError:
            continue        # a cluster came out empty: try another start

        cost = ((v - centroids[labels]) ** 2).sum()
        if best is None or cost < best[0]:
            best = (cost, labels)

    if best is None:
        raise RuntimeError(f"k-means found no {k} non-empty clusters in {KMEANS_RUNS} runs")
    return best[1]


def absorb_small(x, labels, min_size):
    """Tiny clusters are usually one person split off (a laugh, a cough), not a new person:
    move their windows to the most similar big cluster."""
    ids, counts = np.unique(labels, return_counts=True)
    big = ids[counts >= min_size]

    if len(big) == 0 or len(big) == len(ids):
        return labels

    centroids = l2norm(np.stack([x[labels == k].mean(axis=0) for k in big]))
    small = ~np.isin(labels, big)
    labels = labels.copy()
    labels[small] = big[(x[small] @ centroids.T).argmax(axis=1)]
    return labels


def by_first_appearance(labels):
    """Renumber so SPK_0 is whoever speaks first, SPK_1 the next new voice, ..."""
    order = {}
    for k in labels:
        order.setdefault(int(k), len(order))
    return np.array([order[int(k)] for k in labels])


def centroid_similarity(x, labels):
    """Cosine similarity between speaker centroids: the main tool for tuning the clustering."""
    ids = np.unique(labels)
    c = l2norm(np.stack([x[labels == k].mean(axis=0) for k in ids]))
    return c @ c.T



# Timeline

def label_segments(windows, labels, regions):
    """Each window owns the time closest to its centre: boundaries at midpoints between centres."""
    by_region = {}
    for w, label in zip(windows, labels):
        by_region.setdefault(w.region, []).append((w, int(label)))

    out = []
    for r_idx, items in sorted(by_region.items()):
        region = regions[r_idx]
        for i, (w, label) in enumerate(items):
            start = region.start_ms if i == 0 else (items[i - 1][0].center_ms + w.center_ms) // 2
            end = region.end_ms if i == len(items) - 1 else (w.center_ms + items[i + 1][0].center_ms) // 2
            out.append(Turn(start, end, f"SPK_{label}"))
    return out


def merge_adjacent(turns, gap_ms=MERGE_GAP_MS):
    out = []
    for t in turns:
        if out and out[-1].speaker == t.speaker and (t.start_ms - out[-1].end_ms) <= gap_ms:
            out[-1] = Turn(out[-1].start_ms, t.end_ms, t.speaker)
        else:
            out.append(t)
    return out


def smooth(turns, min_turn_ms=MIN_TURN_MS, gap_ms=MERGE_GAP_MS):
    """A very short turn glued to a neighbour is usually a mislabelled window (flicker).
    A short turn standing alone between silences ("Evet." on its own) is kept."""
    out = []
    for i, t in enumerate(turns):
        if t.duration_ms < min_turn_ms:
            prev = out[-1] if out else None
            nxt = turns[i + 1] if i + 1 < len(turns) else None

            if prev and (t.start_ms - prev.end_ms) <= gap_ms:
                t = Turn(t.start_ms, t.end_ms, prev.speaker)
            elif nxt and (nxt.start_ms - t.end_ms) <= gap_ms:
                t = Turn(t.start_ms, t.end_ms, nxt.speaker)

        out.append(t)
    return merge_adjacent(out, gap_ms)
