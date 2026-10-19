"""Live chunking: decide during a recording which speech goes to whisper next.

The same rules as pack() (<= 30 s, cut at silences, a long silence always ends a chunk), but
regions arrive one by one and a chunk is only built when whisper-server is free. While whisper is
busy (~25 s per request on the Pi) speech piles up, so chunks fill up by themselves when it falls
behind. Speech that runs past 30 s without a pause is cut while it is still going ("forced"). Whisper
drops the last 1-3 s of a chunk that ends mid-sentence, so a forced cut can be:
    "hard"     at 30 s
    "quiet"    at the quietest point of the last 5 s (a breath between words)
    "overlap"  at 30 s, and the next chunk starts again 3-6 s earlier, at the quietest point there:
               the tail is heard twice and the words up to the next chunk's start come from the first
"""
from pipeline.config import SttConfig
from pipeline.stt import QUIET_SEARCH_MS, Chunk, quiet_point
from pipeline.vad import SpeechRegion

CUTS = ("hard", "quiet", "overlap")
OVERLAP_MS = (6000, 3000)               # "overlap": the next chunk starts 6-3 s before the cut


class LiveChunker:

    def __init__(self, config: SttConfig, min_send_ms=0, cut="hard", wav=None):
        """min_send_ms: at a pause, send what is there once it is this long, without waiting
        for the chunk to fill up. 0 = send at the first pause while whisper is free.
        cut: how speech without a pause is cut (CUTS); "quiet" and "overlap" read the audio from wav."""
        if cut not in CUTS:
            raise ValueError(f"cut must be one of {CUTS}, got {cut!r}")
        if cut != "hard" and wav is None:
            raise ValueError(f'cut="{cut}" needs the wav')
        self.max_chunk = config.max_chunk_ms
        self.max_gap = config.max_gap_ms
        self.min_send = min_send_ms
        self.cut = cut
        self.wav = wav
        self.pending: list[SpeechRegion] = []      # closed regions, not (fully) sent yet
        self.open_start = None                     # start of the speech still going on, ms
        self.sent_until = 0                        # everything before this ms is sent
        self.index = 0

    def add_region(self, region: SpeechRegion) -> None:
        """A region the VAD has closed."""
        if region.end_ms > self.sent_until:
            self.pending.append(region)

    def set_open(self, start_ms) -> None:
        """Speech that has started but not ended yet (None: silence now)."""
        self.open_start = start_ms

    @property
    def empty(self) -> bool:
        return not self.pending

    def next_chunk(self, now_ms, final=False) -> Chunk | None:
        """Called whenever whisper is free. final: the recording has stopped, send everything."""
        spans = [(max(r.start_ms, self.sent_until), r.end_ms) for r in self.pending]
        open_start = None if self.open_start is None else max(self.open_start, self.sent_until)

        if not spans:
            if open_start is not None and now_ms - open_start >= self.max_chunk:
                return self.forced(open_start)      # no pause for 30 s
            return None

        start, end = spans[0]
        if end - start > self.max_chunk:
            return self.forced(start)               # closed while whisper was busy
        for s, e in spans[1:]:
            if s - end > self.max_gap:
                return self.emit(start, end, "gap")
            if e - start > self.max_chunk:
                return self.emit(start, end, "full")
            end = e

        # every closed region fits: wait for more, unless nothing more can join this chunk
        if final:
            return self.emit(start, end, "end")
        if open_start is not None:
            if open_start - end > self.max_gap:
                return self.emit(start, end, "gap")
            if now_ms - start > self.max_chunk:
                return self.emit(start, end, "full")        # the speech going on no longer fits
        elif now_ms - end > self.max_gap:
            return self.emit(start, end, "gap")
        if end - start >= self.min_send:
            return self.emit(start, end, "early")
        return None

    def forced(self, start) -> Chunk:
        """30 s of speech from start without a pause long enough for the VAD."""
        limit = start + self.max_chunk
        if self.cut == "quiet":
            return self.emit(start, quiet_point(self.wav, limit - QUIET_SEARCH_MS, limit), "forced")
        chunk = self.emit(start, limit, "forced")
        if self.cut == "overlap":
            self.sent_until = quiet_point(self.wav, limit - OVERLAP_MS[0], limit - OVERLAP_MS[1])
        return chunk

    def emit(self, start, end, reason) -> Chunk:
        self.sent_until = end
        self.pending = [r for r in self.pending if r.end_ms > end]
        chunk = Chunk(self.index, start, end, reason)
        self.index += 1
        return chunk
