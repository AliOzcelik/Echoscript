// Runs on the audio thread. Averages the mic signal down to 16 kHz mono int16,
// the format /ws/audio expects, and posts it in ~100 ms chunks.
class PcmDownsampler extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000;   // input samples per output sample (sampleRate is a worklet global)
    this.pos = 0;
    this.sum = 0;
    this.count = 0;
    this.out = [];
  }

  process(inputs) {
    const input = inputs[0][0];        // first channel of the first input
    if (!input) return true;

    for (const x of input) {
      this.sum += x;
      this.count++;
      this.pos++;
      if (this.pos >= this.ratio) {
        this.pos -= this.ratio;
        this.out.push(Math.max(-1, Math.min(1, this.sum / this.count)) * 0x7fff);
        this.sum = 0;
        this.count = 0;
      }
    }

    if (this.out.length >= 1600) {
      const pcm = Int16Array.from(this.out);
      this.out = [];
      this.port.postMessage(pcm.buffer, [pcm.buffer]);
    }
    return true;
  }
}

registerProcessor('pcm-downsampler', PcmDownsampler);
