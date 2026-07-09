import webrtcvad
from collections import deque


# `webrtcvad` is picky. It accepts *only* 16-bit mono PCM, *only* at 8/16/32/48 kHz, and *only* in frames of exactly 10/20/30 ms. At 16 kHz, 30 ms = 480 samples = **960 bytes**. If you hand it anything else >

# SAMPLE_WIDTH = 2        # bytes/sample -> 16-bit signed PCM ("int16")
# FRAME_MS = 30           # webrtcvad ONLY accepts 10, 20, or 30 ms frames

# Derived constants used everywhere below:
# BYTES_PER_FRAME = int(SAMPLE_RATE * FRAME_MS / 1000) * SAMPLE_WIDTH  # = 960 bytes
# RECORDINGS_DIR = Path("recordings")



# start automatically when someone speaks
# stop after sufficient silence
class SpeechSegment:

        def __init__(self, sample_rate=16000, sample_width=2, frame_ms=30, aggressiveness=2, preroll_ms=300, end_silence_ms=1200, min_segment_ms=8000, max_segment_ms=90000):
            # Higer aggressiveness eager to call audio 'not speech'
            self.sample_rate = sample_rate
            self.bytes_per_frame = int(sample_rate * frame_ms / 1000) * sample_width
            self.vad = webrtcvad.Vad(aggressiveness)

            # convert the millisecond knobs into a count of 30 ms frames
            self.end_silence_frames = end_silence_ms // frame_ms
            self.min_segment_frames = min_segment_ms // frame_ms
            self.max_segment_frames = max_segment_ms // frame_ms

            # a fixed-size window of the most recent frames while IDLE = the pre-roll
            self.preroll = deque(maxlen=preroll_ms // frame_ms)

            self.leftover = b""      # bytes received but not yet a full 30 ms frame
            self.recording = False   # which state we're in
            self.segment = []        # list[bytes] — frames of the current segment
            self.silence_run = 0     # consecutive silent frames while RECORDING


        # input: bytes
        # output: list[bytes]
        def add_audio(self, pcm_bytes):
            self.leftover += pcm_bytes
            completed = []

            # slice the byte stream into 30 ms frames for the vad
            while len(self.leftover) >= self.bytes_per_frame:
                frame = self.leftover[:self.bytes_per_frame]
                self.leftover = self.leftover[self.bytes_per_frame:]
                segment = self.preprocess_frame(frame)
                if segment is not None:
                    completed.append(segment)

            return completed


        def preprocess_frame(self, frame):
            is_speech = self.vad.is_speech(frame, self.sample_rate)

            if not self.recording:
                # IDLE: remember recent frames; a speech frame triggers recording.
                self.preroll.append(frame)
                if is_speech:
                    self.recording = True
                    self.segment = list(self.preroll)  # prepend the pre-roll
                    self.preroll.clear()
                    self.silence_run = 0
                return None

            # RECORDING: keep everything, count trailing silence.
            self.segment.append(frame)
            self.silence_run = 0 if is_speech else self.silence_run + 1
            if self.silence_run >= self.end_silence_frames:
                return self.close_segment()      # a natural pause ended the segment
            if len(self.segment) >= self.max_segment_frames:
                return self.close_segment()      # someone monologued past the cap
            return None


        def close_segment(self):
            frames = self.segment
            self.segment = []
            self.recording = False
            self.silence_run = 0
            self.preroll.clear()
            if len(frames) < self.min_segment_frames:
                return None                       # too short -> not worth the pipeline
            return b"".join(frames)

        def flush(self):
            # When the socket closes mid-speech, emit the tail if it's long enough.
            if self.recording and len(self.segment) >= self.min_segment_frames:
                return b"".join(self.segment)
            return None
