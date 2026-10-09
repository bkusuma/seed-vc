"""Lightweight real-time I/O for Seed-VC v3: echo cancellation, mic gain control, duplex stream.

Signal chain per block (device sample rate, mono):

    mic -> EchoCanceller(far = what we played) -> AutomaticGainControl -> 16 kHz
        -> StreamingVoiceConverter -> output rate -> speakers (and AEC far-end reference)

Everything is numpy/torch; ``sounddevice`` (PortAudio) is imported only when a
real device stream is opened, so ``simulate_stream`` works on headless machines.

Echo cancellation matters for voice conversion beyond politeness: with speakers
instead of headphones, the converted voice leaks back into the mic and is
converted again, which builds into a feedback loop.
"""
import math
import time
from collections import deque

import numpy as np
from scipy.signal import lfilter

from modules.v3.audio import StreamingResampler


def _db(power):
    return 10.0 * np.log10(power + 1e-12)


def default_frame_size(sample_rate):
    """10 ms when that is an integer number of samples, else 20 ms (e.g. 22.05 kHz), so frames
    always tile the 20 ms block grid used by the streaming engine."""
    return sample_rate // 100 if sample_rate % 100 == 0 else sample_rate // 50


class EchoCanceller:
    """Two-path partitioned-block frequency-domain NLMS acoustic echo canceller.

    * ``tail_ms`` of echo path modelled with ``ceil(tail * sr / N)`` partitions of
      N = ``frame_size`` samples (default 10 ms, or 20 ms where 10 ms is fractional); 2N-point FFTs.
    * NLMS normalisation per bin by the far-end power across the whole filter
      span (sum over partitions), so speech onsets cannot cause oversized steps.
    * Two echo-path models (Ochiai et al., 1977; also Speex MDF's TWO_PATH): a
      *background* filter adapts whenever the far end is active, a *foreground*
      filter produces the output. The background is copied to the foreground
      when its error is clearly lower, and reset from it when it diverges (e.g.
      during double talk). This gives fast convergence without a hard
      double-talk detector, whose false alarms would stall adaptation.
    * Geigel double-talk gate on the background (``dtd_threshold``; None disables).
    * Non-linear processor (``nlp_strength``; 0 disables): attenuates frames in
      proportion to how much of the mic energy the echo estimate explains.

    Defaults were picked on a synthetic 44.1 kHz test (speech far end, 30 ms bulk
    delay + 120 ms exponentially decaying room response at -6 dB, -65 dBFS noise):
    ~22 dB steady-state ERLE, ~24 dB near-end SNR during double talk, ~2% of one
    CPU core (tests/v3/test_realtime_io.py). Real rooms with loudspeaker
    non-linearities will do worse; headphones remain the robust option.
    * ``bulk_delay_ms`` delays the far-end reference so the filter span covers the
      device round trip (playback -> air -> capture) plus the room tail.

    ``process`` lengths must be multiples of ``frame_size``; blocks that are
    multiples of 20 ms always are.
    """

    def __init__(self, sample_rate, frame_size=None, tail_ms=200.0, mu=1.0, regularization=1e-3,
                 copy_ratio_db=1.0, reset_ratio_db=6.0, energy_smoothing=0.8, dtd_threshold=0.5,
                 dtd_hangover_ms=150.0, far_activity_dbfs=-55.0, bulk_delay_ms=0.0, nlp_strength=0.3):
        self.sr = int(sample_rate)
        self.N = int(frame_size or default_frame_size(self.sr))
        self.P = max(1, math.ceil(tail_ms / 1000.0 * self.sr / self.N))
        self.mu = mu
        self.regularization = regularization
        self.copy_ratio = 10 ** (-copy_ratio_db / 10.0)
        self.reset_ratio = 10 ** (reset_ratio_db / 10.0)
        self.beta = energy_smoothing
        self.dtd_threshold = dtd_threshold
        self.hangover = max(1, int(round(dtd_hangover_ms / 1000.0 * self.sr / self.N)))
        self.far_activity_dbfs = far_activity_dbfs
        self.nlp_strength = nlp_strength
        self.bulk_delay = int(round(bulk_delay_ms / 1000.0 * self.sr))
        self.reset()

    def reset(self):
        N, P = self.N, self.P
        self.W_bg = np.zeros((P, N + 1), dtype=np.complex128)
        self.W_fg = np.zeros((P, N + 1), dtype=np.complex128)
        self.X = np.zeros((P, N + 1), dtype=np.complex128)
        self.x_prev = np.zeros(N)
        self.far_hist = np.zeros(P * N + N)
        self.delay_line = np.zeros(self.bulk_delay)
        self.dt_hold = 0
        self.constrain_idx = 0
        self.nlp_gain = 1.0
        self.e_fg_energy = 0.0
        self.e_bg_energy = 0.0
        self.copies = 0
        self.resets = 0

    def _delay(self, far):
        if self.bulk_delay == 0:
            return far
        buf = np.concatenate([self.delay_line, far])
        self.delay_line = buf[-self.bulk_delay:]
        return buf[: len(far)]

    def _estimate(self, W):
        return np.fft.irfft((W * self.X).sum(0), n=2 * self.N)[self.N:]

    def _frame(self, d, x):
        N = self.N
        self.X[1:] = self.X[:-1]
        self.X[0] = np.fft.rfft(np.concatenate([self.x_prev, x]))
        self.x_prev = x
        y_fg, y_bg = self._estimate(self.W_fg), self._estimate(self.W_bg)
        e_fg, e_bg = d - y_fg, d - y_bg

        b = self.beta
        self.e_fg_energy = b * self.e_fg_energy + (1 - b) * float(np.sum(e_fg ** 2))
        self.e_bg_energy = b * self.e_bg_energy + (1 - b) * float(np.sum(e_bg ** 2))
        far_active = _db(np.mean(x ** 2)) > self.far_activity_dbfs
        if far_active and self.e_bg_energy < self.copy_ratio * self.e_fg_energy:
            self.W_fg[:] = self.W_bg
            e_fg, y_fg = e_bg, y_bg
            self.e_fg_energy = self.e_bg_energy
            self.copies += 1
        elif self.e_bg_energy > self.reset_ratio * self.e_fg_energy:
            self.W_bg[:] = self.W_fg
            e_bg = e_fg
            self.e_bg_energy = self.e_fg_energy
            self.resets += 1

        self.far_hist = np.concatenate([self.far_hist[N:], x])
        if self.dtd_threshold is not None:
            far_peak = np.abs(self.far_hist).max()
            if far_peak > 0 and np.abs(d).max() > self.dtd_threshold * far_peak:
                self.dt_hold = self.hangover
            elif self.dt_hold > 0:
                self.dt_hold -= 1

        if far_active and self.dt_hold == 0:
            E = np.fft.rfft(np.concatenate([np.zeros(N), e_bg]))
            psd = (np.abs(self.X) ** 2).sum(0)
            denom = psd + self.regularization * psd.mean() + 1e-12
            self.W_bg += self.mu * np.conj(self.X) * (E / denom)[None]
            # gradient constraint: partition 0 every frame + one more round-robin (AUMDF-style)
            for p in {0, self.constrain_idx}:
                w = np.fft.irfft(self.W_bg[p], n=2 * N)
                w[N:] = 0
                self.W_bg[p] = np.fft.rfft(w)
            self.constrain_idx = (self.constrain_idx + 1) % self.P

        out = e_fg
        if self.nlp_strength > 0:
            ratio = np.clip(np.mean(y_fg ** 2) / (np.mean(d ** 2) + 1e-12), 0.0, 1.0)
            target = 1.0 - self.nlp_strength * ratio
            out = out * np.linspace(self.nlp_gain, target, N, endpoint=False)
            self.nlp_gain = target
        return out

    def process(self, mic, far):
        mic = np.asarray(mic, dtype=np.float64).reshape(-1)
        far = self._delay(np.asarray(far, dtype=np.float64).reshape(-1))
        if len(mic) != len(far) or len(mic) % self.N:
            raise ValueError(f"mic/far must have equal length, a multiple of {self.N}")
        out = np.empty_like(mic)
        for i in range(0, len(mic), self.N):
            out[i:i + self.N] = self._frame(mic[i:i + self.N], far[i:i + self.N])
        return out.astype(np.float32)


class AutomaticGainControl:
    """Mic gain normalisation with a noise gate and a soft limiter.

    Level is tracked per 10 ms frame (20 ms at rates like 22.05 kHz; fast attack, slow release). The gain moves
    toward ``target_dbfs - level`` (clamped to [min_gain_db, max_gain_db]); it
    falls quickly and rises slowly so loud onsets are not over-amplified. While
    the gate is closed (level below ``gate_dbfs`` for longer than ``gate_hold_ms``)
    the gain is frozen and the signal attenuated by ``gate_attenuation_db``, so
    background noise is not boosted into the converter. A tanh soft limiter keeps
    peaks below 0 dBFS.
    """

    def __init__(self, sample_rate, target_dbfs=-20.0, max_gain_db=30.0, min_gain_db=-20.0, attack_ms=10.0,
                 release_ms=300.0, gain_down_ms=50.0, gain_up_ms=800.0, gate_dbfs=-50.0, gate_hold_ms=250.0,
                 gate_attenuation_db=-30.0, limiter_dbfs=-1.0):
        self.sr = int(sample_rate)
        self.frame = default_frame_size(self.sr)
        frame_ms = 1000.0 * self.frame / self.sr
        coef = lambda ms: math.exp(-frame_ms / max(ms, 1e-3))
        self.a_attack, self.a_release = coef(attack_ms), coef(release_ms)
        self.a_down, self.a_up = coef(gain_down_ms), coef(gain_up_ms)
        self.target, self.max_gain, self.min_gain = target_dbfs, max_gain_db, min_gain_db
        self.gate_dbfs, self.gate_attenuation_db = gate_dbfs, gate_attenuation_db
        self.hold_frames = max(1, int(round(gate_hold_ms / frame_ms)))
        self.limit = 10 ** (limiter_dbfs / 20.0)
        self.reset()

    def reset(self):
        self.level_db = self.target
        self.gain_db = 0.0
        self.gate_open = False
        self.hold_left = 0
        self.prev_lin = 10 ** (self.gate_attenuation_db / 20.0)

    def _soft_limit(self, y):
        thr = self.limit
        mag = np.abs(y)
        over = mag > thr
        if np.any(over):
            y = y.copy()
            y[over] = np.sign(y[over]) * (thr + (1 - thr) * np.tanh((mag[over] - thr) / (1 - thr)))
        return y

    def process(self, x):
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if len(x) % self.frame:
            raise ValueError(f"length must be a multiple of {self.frame} samples")
        out = np.empty_like(x)
        for i in range(0, len(x), self.frame):
            f = x[i:i + self.frame]
            rms_db = _db(np.mean(f ** 2))
            a = self.a_attack if rms_db > self.level_db else self.a_release
            self.level_db = a * self.level_db + (1 - a) * rms_db
            if rms_db > self.gate_dbfs:
                self.gate_open, self.hold_left = True, self.hold_frames
            elif self.hold_left > 0:
                self.hold_left -= 1
                self.gate_open = self.hold_left > 0
            if self.gate_open:
                desired = float(np.clip(self.target - self.level_db, self.min_gain, self.max_gain))
                a = self.a_down if desired < self.gain_db else self.a_up
                self.gain_db = a * self.gain_db + (1 - a) * desired
            applied = self.gain_db + (0.0 if self.gate_open else self.gate_attenuation_db)
            lin = 10 ** (applied / 20.0)
            out[i:i + self.frame] = f * np.linspace(self.prev_lin, lin, self.frame, endpoint=False)
            self.prev_lin = lin
        return self._soft_limit(out).astype(np.float32)


def _import_sounddevice():
    try:
        import sounddevice as sd
    except OSError as e:  # PortAudio missing
        raise RuntimeError("sounddevice could not load PortAudio (e.g. `apt install libportaudio2`); "
                           "use simulate_stream / realtime_v3.py --simulate on headless machines") from e
    return sd


class RealtimeVCStream:
    """Duplex device stream around a ``StreamingVoiceConverter``.

    ``samplerate`` defaults to the model rate (44.1 kHz), which avoids output
    resampling. Inference runs inside the PortAudio callback (as in
    real-time-gui.py), so per-block compute must stay below the block time.
    """

    def __init__(self, engine, samplerate=None, input_device=None, output_device=None, channels=1,
                 aec=True, agc=True, mute_when_gated=False, aec_kwargs=None, agc_kwargs=None, latency="low"):
        self.engine = engine
        self.sr = int(samplerate or engine.output_sr)
        if self.sr % 50:
            raise ValueError("device sample rate must be a multiple of 50 Hz")
        self.block = engine.n_block * self.sr // 50
        self.to_16k = StreamingResampler(self.sr, 16000)
        self.from_model = StreamingResampler(engine.output_sr, self.sr) if engine.output_sr != self.sr else None
        self.aec = EchoCanceller(self.sr, **(aec_kwargs or {})) if aec else None
        self.agc = AutomaticGainControl(self.sr, **(agc_kwargs or {})) if agc else None
        self.mute_when_gated = mute_when_gated and agc
        self.input_device, self.output_device = input_device, output_device
        self.channels = channels
        self.latency = latency
        self.far = np.zeros(self.block, dtype=np.float32)
        self.callback_ms = deque(maxlen=200)
        self.status_count = 0
        self.stream = None

    @property
    def resampler_delay_ms(self):
        delay = self.to_16k.delay_seconds + (self.from_model.delay_seconds if self.from_model else 0.0)
        return 1000.0 * delay

    def process(self, mic_block):
        """Device-rate mono block in -> device-rate mono block out."""
        x = np.asarray(mic_block, dtype=np.float32).reshape(-1)
        if self.aec is not None:
            x = self.aec.process(x, self.far)
        if self.agc is not None:
            x = self.agc.process(x)
        y = self.engine.process_block(self.to_16k(x))
        if self.from_model is not None:
            y = self.from_model(y).cpu().numpy()
        y = np.asarray(y, dtype=np.float32)
        if self.mute_when_gated and not self.agc.gate_open:
            y = np.zeros_like(y)
        self.far = y
        return y

    def _callback(self, indata, outdata, frames, time_info, status):
        start = time.perf_counter()
        if status:
            self.status_count += 1
        try:
            y = self.process(indata.mean(axis=1))
            outdata[:] = np.repeat(y[:, None], outdata.shape[1], axis=1)
        except Exception as e:  # never let an exception kill the audio thread silently
            outdata.fill(0)
            print(f"[realtime_io] block failed: {e!r}")
        self.callback_ms.append(1000.0 * (time.perf_counter() - start))

    def start(self):
        sd = _import_sounddevice()
        self.stream = sd.Stream(samplerate=self.sr, blocksize=self.block, dtype="float32",
                                channels=(self.channels, self.channels), device=(self.input_device, self.output_device),
                                latency=self.latency, callback=self._callback)
        self.stream.start()
        return self

    def device_latency_ms(self):
        if self.stream is None:
            return None
        return 1000.0 * float(sum(self.stream.latency)) + self.resampler_delay_ms

    def stop(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def run(self, duration=None, report_every=5.0):
        self.start()
        print(f"[realtime_io] {self.sr} Hz, block {self.block} samples; latency report:",
              self.engine.latency_report(self.device_latency_ms()))
        t0 = last = time.time()
        try:
            while duration is None or time.time() - t0 < duration:
                time.sleep(0.1)
                if time.time() - last >= report_every and self.callback_ms:
                    last = time.time()
                    ms = np.array(self.callback_ms)
                    print(f"[realtime_io] callback mean {ms.mean():.1f} ms, p95 {np.percentile(ms, 95):.1f} ms "
                          f"(budget {1000 * self.block / self.sr:.0f} ms), status events {self.status_count}")
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()


def simulate_stream(stream, input_wave, echo_ir=None):
    """Run ``RealtimeVCStream.process`` over a file, block by block, without audio hardware.

    echo_ir: optional FIR (device rate) for an acoustic path from the output back
    into the mic, closing the feedback loop to exercise the echo canceller.
    Returns (output, per_block_seconds).
    """
    x = np.asarray(input_wave, dtype=np.float32).reshape(-1)
    n_blocks = len(x) // stream.block
    zi = np.zeros(len(echo_ir) - 1) if echo_ir is not None else None
    outs, times = [], []
    prev_out = np.zeros(stream.block, dtype=np.float32)
    for i in range(n_blocks):
        mic = x[i * stream.block:(i + 1) * stream.block]
        if echo_ir is not None:
            echo, zi = lfilter(echo_ir, [1.0], prev_out, zi=zi)
            mic = mic + echo.astype(np.float32)
        t = time.perf_counter()
        prev_out = stream.process(mic)
        times.append(time.perf_counter() - t)
        outs.append(prev_out)
    return (np.concatenate(outs) if outs else np.zeros(0, dtype=np.float32)), np.array(times)
