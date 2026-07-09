from dataclasses import dataclass


# One speaker actually saying something
# Speaker 1, from 3.4s to 7.1s
@dataclass
class Utterance:
    speaker: str
    start: float
    end: float
    text: str


# Takes the diarization turns and the transcript, and stitch them together
class Merge:

    def __init__(self, turns, transcript):

        self.turns = turns # list[SpeakerTurn] form diarize
        self.transcript = transcript # from transcript file

    # which speaker was talking during [start, end]
    def speaker_for(self, start, end):
        best_speaker = None
        best_overlap = 0.0
        for t in self.turns:
            overlap = min(end, t.end) - max(start, t.start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = t.speaker

        if best_speaker is not None:
            return best_speaker

        # The word fell in a gap with no overlapping turn -> use the nearest one.
        mid = (start + end) / 2
        nearest = min(self.turns, key=lambda t: min(abs(mid - t.start), abs(mid - t.end)))
        return nearest.speaker


    def merge(self):

        segs = self.transcript.segments

        # No diarization, only one person
        if not self.turns:
            text = self.transcript.text
            if not text:
                return []
            return [Utterance("Speaker_00", segs[0].start, segs[0].end, text)]

        utterances = []
        current = None

        for seg in segs:
            # Prefer per-word timing; fall back to the whole segment if needed.
            units = seg.words if seg.words else [seg]
            for u in units:
                speaker = self.speaker_for(u.start, u.end)
                if current is not None and current.speaker == speaker:
                    # Same speaker keeps talking -> extend the current utterance.
                    current.text += " " + u.text
                    current.end = u.end
                else:
                    # Speaker changed -> close the old utterance, start a new one.
                    if current is not None:
                        utterances.append(current)
                    current = Utterance(speaker, u.start, u.end, u.text)

        if current is not None:
            utterances.append(current)

        return utterances


# mm:ss formatting so the output reads like a transcript, not raw seconds.
def _timestamp(seconds):
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:d}:{secs:02d}"


# Render utterances into a readable, speaker-labeled transcript string.
def render(utterances):
    lines = []
    for u in utterances:
        lines.append(f"[{_timestamp(u.start)}] {u.speaker}: {u.text}")
    return "\n".join(lines)
