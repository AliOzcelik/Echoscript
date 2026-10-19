"""Live mode simulation: replays a recording's VAD regions on a virtual clock through LiveChunker,
with whisper-server's time per request from a cost model fitted on real runs. Instant, no STT.

Reports how far the text lags behind the speech and how long after stop the transcript is done,
next to the offline pipeline (pack, everything after stop).

    python -m eval.live_sim recordings/*/
    python -m eval.live_sim recordings/oscar_da_bizi_gorecek_mi --min-send 0 10 20
    python -m eval.live_sim recordings/*/ --min-send 0 --cut hard quiet overlap
"""
import argparse
import json
import statistics
from pathlib import Path

import soundfile as sf

from pipeline.config import load_config
from pipeline.live import CUTS, LiveChunker
from pipeline.stt import pack
from pipeline.vad import SpeechRegion

STEP_MS = 100
# seconds per request = COST_A + COST_B * chunk seconds: fitted on the 188 chunks of runs/A (sd 2.8 s)
COST_A = 21.9
COST_B = 0.19


def cost_s(chunk) -> float:
    return COST_A + COST_B * chunk.duration_ms / 1000


def simulate(regions, total_ms, cfg, min_send_ms=0, cut="hard", wav=None):
    """-> [(chunk, sent_ms, done_ms)]. A region is open from its (unpadded) start until the VAD
    closes it: min_silence after the speech ended, i.e. pad before its padded end."""
    stt, vad = cfg.stt, cfg.vad
    chunker = LiveChunker(stt, min_send_ms, cut, wav)
    closes = [(r.end_ms - vad.pad_ms + vad.min_silence_ms, r) for r in regions]
    nxt = 0                 # next region to close
    free_at = 0             # whisper-server is busy until then
    out = []
    t = 0
    while True:
        final = t >= total_ms       # stop pressed: the VAD flushes, every region is closed
        while nxt < len(closes) and (final or closes[nxt][0] <= t):
            chunker.add_region(closes[nxt][1])
            nxt += 1
        speaking = nxt < len(regions) and regions[nxt].start_ms + vad.pad_ms <= t
        chunker.set_open(regions[nxt].start_ms if speaking and not final else None)

        if t >= free_at:
            chunk = chunker.next_chunk(t, final)
            if chunk is not None:
                free_at = t + round(cost_s(chunk) * 1000)
                out.append((chunk, t, free_at))
                continue            # whisper is busy now; same t, nothing more to send
        if final and chunker.empty and t >= free_at:
            return out
        t += STEP_MS


def report(name, runs, total_ms):
    chunks = [c for c, _, _ in runs]
    # a word waits from when it was spoken until its chunk's text is back; mid-chunk = average word
    delay = [(done - (c.start_ms + c.end_ms) / 2) / 1000 for c, _, done in runs]
    q = statistics.quantiles(delay, n=10)
    after_stop = (runs[-1][2] - total_ms) / 1000
    fill = sum(c.duration_ms for c in chunks) / (len(chunks) * 30000)
    reasons = {r: sum(c.reason == r for c in chunks) for r in ("early", "full", "gap", "forced", "end")}
    print(f"  {name:16s} {len(chunks):3d} chunks  fill {fill:4.0%}  delay p50 {q[4] / 60:4.1f} min  "
          f"p90 {q[8] / 60:4.1f} min  max {max(delay) / 60:4.1f} min  done {after_stop / 60:4.1f} min after stop  "
          + " ".join(f"{r} {n}" for r, n in reasons.items() if n))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("recs", type=Path, nargs="+", help="folders with audio.wav and vad.json")
    ap.add_argument("--min-send", type=float, nargs="+", default=[0, 15, 25], help="LiveChunker min_send, s")
    ap.add_argument("--cut", nargs="+", choices=CUTS, default=["hard"], help="how speech without a pause is cut")
    args = ap.parse_args()
    cfg = load_config()

    for rec in args.recs:
        regions = [SpeechRegion(**r) for r in json.loads((rec / "vad.json").read_text())["regions"]]
        total_ms = int(sf.info(str(rec / "audio.wav")).duration * 1000)
        offline = pack(regions, cfg.stt)
        print(f"{rec.name}  ({total_ms / 60000:.1f} min)")
        print(f"  {'offline (A)':16s} {len(offline):3d} chunks  done "
              f"{sum(cost_s(c) for c in offline) / 60:4.1f} min after stop")
        for cut in args.cut:
            for s in args.min_send:
                report(f"{cut} min {s:g} s", simulate(regions, total_ms, cfg, int(s * 1000), cut, rec / "audio.wav"), total_ms)


if __name__ == "__main__":
    main()
