"""Merge: every word gets the speaker of the turn it overlaps most; words -> utterances.

Assigning whole words to speakers means a speaker change never falls inside a word. Diarization
still places changes 0.5-1 s off (1.5 s windows), which cuts sentences ("demek | ki varız"),
so each change then moves to the nearest break: a pause, or a sentence end.

Whisper folds the pause after a sentence into its last word ("hocam?" lasting 2 s), which then
reaches into the next speaker's turn and was taken from the speaker who said it. Speakers and
breaks are therefore decided on word times capped at MAX_WORD_MS; the transcript keeps Whisper's.
"""
from bisect import bisect_right
from dataclasses import dataclass, replace

from pipeline.diar import Turn
from pipeline.stt import Word


MAX_PAUSE_MS = 2000     # same speaker, but a longer pause -> new utterance (paragraph)
SNAP_MS = 2000          # a speaker change moves at most this far to reach a break
MIN_PAUSE_MS = 150      # a gap between words this long is a break
MAX_WORD_MS = 1500      # longer is a folded pause, not speech (oscar: 99% of mid-sentence words <= 1.14 s)
# Whisper's segment times leave no gap between words (a pause is folded into the word before it),
# so its punctuation is usually the only sign of where one speaker stopped
SENTENCE_END = (".", "?", "!", "…")


@dataclass(frozen=True)
class Utterance:
    speaker: str
    words: tuple    # tuple of Word, in time order

    @property
    def start_ms(self):
        return self.words[0].start_ms

    @property
    def end_ms(self):
        return self.words[-1].end_ms

    @property
    def text(self):
        return " ".join(w.text for w in self.words)

    def to_dict(self):
        return {
            "speaker": self.speaker,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "text": self.text,
            "words": [w.to_dict() for w in self.words],
        }



def merge(words, turns):
    starts = [t.start_ms for t in turns]
    timed = [capped(w) for w in words]
    speakers = snap([speaker_of(w, turns, starts) for w in timed], timed)
    utterances = []
    current = []
    speaker = None

    for w, s in zip(words, speakers):
        if current and (s != speaker or (w.start_ms - current[-1].end_ms) > MAX_PAUSE_MS):
            utterances.append(Utterance(speaker, tuple(current)))     # new speaker, or a long pause
            current = []

        current.append(w)
        speaker = s

    if current:
        utterances.append(Utterance(speaker, tuple(current)))

    return utterances


def snap(speakers, words):
    """Move every speaker change to the nearest break within SNAP_MS; without one it stays."""
    speakers = list(speakers)
    changes = [k for k in range(1, len(words)) if speakers[k] != speakers[k - 1]]   # k: first word of the new speaker

    for n, b in enumerate(changes):
        # stay between the neighbouring changes, so no utterance loses all its words
        lo = changes[n - 1] + 1 if n > 0 else 1
        hi = changes[n + 1] if n + 1 < len(changes) else len(words)
        t = words[b].start_ms
        breaks = [k for k in range(lo, hi) if is_break(words, k) and break_distance(words, k, t) <= SNAP_MS]
        if not breaks:
            continue

        k = min(breaks, key=lambda k: break_distance(words, k, t))
        if k < b:
            speakers[k:b] = [speakers[b]] * (b - k)             # the new speaker starts earlier
        else:
            speakers[b:k] = [speakers[b - 1]] * (k - b)         # the old speaker finishes the sentence
        changes[n] = k

    return speakers


def break_distance(words, k, t):
    """How far time t is from the break before word k: from the nearer edge of the gap
    (word k-1's end .. word k's start), so a break just after the change is near."""
    gap_start, gap_end = words[k - 1].end_ms, words[k].start_ms
    if gap_start <= t <= gap_end:
        return 0
    return min(abs(gap_start - t), abs(gap_end - t))


def capped(word):
    if word.end_ms - word.start_ms <= MAX_WORD_MS:
        return word
    return replace(word, end_ms=word.start_ms + MAX_WORD_MS)


def is_break(words, k):
    """Is there a break between word k-1 and word k?"""
    prev = words[k - 1]
    return (words[k].start_ms - prev.end_ms) >= MIN_PAUSE_MS or prev.text.endswith(SENTENCE_END)


def speaker_of(word, turns, starts):
    if not turns:
        return "SPK_?"

    # Turns are sorted and never overlap, so the ones overlapping this word sit
    # just before index i: walk back from there while they still reach into the word.
    i = bisect_right(starts, word.end_ms)
    best = None
    best_overlap = 0

    j = i - 1
    while j >= 0 and turns[j].end_ms > word.start_ms:
        overlap = min(word.end_ms, turns[j].end_ms) - max(word.start_ms, turns[j].start_ms)
        if overlap > best_overlap:
            best = turns[j]
            best_overlap = overlap
        j -= 1

    if best is not None:
        return best.speaker

    # No overlap (word in a gap, or in a region too short for a voiceprint): nearest turn wins
    before = turns[i - 1] if i > 0 else None
    after = turns[i] if i < len(turns) else None

    if before is None:
        return after.speaker
    if after is None:
        return before.speaker
    if (word.start_ms - before.end_ms) <= (after.start_ms - word.end_ms):
        return before.speaker
    return after.speaker
