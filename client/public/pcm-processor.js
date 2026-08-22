/**
 * AudioWorklet processor for streaming PCM audio to the backend.
 *
 * Captures Float32 audio from the microphone at the AudioContext sample rate,
 * resamples to 16kHz mono, converts to Int16 (linear16), and sends chunks
 * to the main thread via MessagePort.
 *
 * Chunk size: 3200 bytes = 1600 samples = 100ms at 16kHz.
 * This is the optimal size for Sarvam's realtime streaming API.
 */
class PCMProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buffer = new Float32Array(0);
    this._chunkSize = 3200; // 1600 samples * 2 bytes = 3200 bytes = 100ms at 16kHz
    this._targetRate = 16000;
    this._sourceRate = sampleRate; // AudioContext sampleRate (e.g., 44100 or 48000)
    this._ratio = this._sourceRate / this._targetRate;
    this._active = true;

    this.port.onmessage = (e) => {
      if (e.data === 'stop') {
        this._active = false;
      }
    };
  }

  process(inputs) {
    if (!this._active) return false;

    const input = inputs[0];
    if (!input || !input.length) return true;

    // Mono: take first channel.
    const channel = input[0];
    if (!channel || !channel.length) return true;

    // Resample to 16kHz using linear interpolation.
    const resampledLength = Math.ceil(channel.length / this._ratio);
    const resampled = new Float32Array(resampledLength);
    for (let i = 0; i < resampledLength; i++) {
      const srcIndex = i * this._ratio;
      const index0 = Math.floor(srcIndex);
      const index1 = Math.min(index0 + 1, channel.length - 1);
      const frac = srcIndex - index0;
      resampled[i] = channel[index0] * (1 - frac) + channel[index1] * frac;
    }

    // Append to buffer.
    const newBuffer = new Float32Array(this._buffer.length + resampled.length);
    newBuffer.set(this._buffer, 0);
    newBuffer.set(resampled, this._buffer.length);
    this._buffer = newBuffer;

    // Emit complete chunks.
    while (this._buffer.length >= this._chunkSize / 2) {
      const chunkSamples = this._chunkSize / 2; // 1600 samples
      const chunk = this._buffer.subarray(0, chunkSamples);
      this._buffer = this._buffer.subarray(chunkSamples);

      // Convert Float32 to Int16 (linear16).
      const int16 = new Int16Array(chunkSamples);
      for (let i = 0; i < chunkSamples; i++) {
        const s = Math.max(-1, Math.min(1, chunk[i]));
        int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
      }

      // Send as ArrayBuffer (transferable).
      this.port.postMessage(int16.buffer, [int16.buffer]);
    }

    return true;
  }
}

registerProcessor('pcm-processor', PCMProcessor);
