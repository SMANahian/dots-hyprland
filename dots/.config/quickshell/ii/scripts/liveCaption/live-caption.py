#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import queue
import signal
import subprocess
import sys
import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr, flush=True)


def run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()


def pactl_default_sink() -> str:
    return run(["pactl", "get-default-sink"])


def pactl_default_source() -> str:
    for line in run(["pactl", "info"]).splitlines():
        if line.startswith("Default Source:"):
            return line.split(":", 1)[1].strip()
    return ""


def pactl_has_source(name: str) -> bool:
    if name.startswith("@") and name.endswith("@"):
        return True
    try:
        out = run(["pactl", "list", "short", "sources"])
    except Exception:
        return False
    for line in out.splitlines():
        cols = line.split("\t")
        if len(cols) >= 2 and cols[1] == name:
            return True
    return False


def pactl_active_sink() -> str:
    # Prefer the sink that currently has audio routed into it (sink-inputs).
    try:
        sink_inputs = run(["pactl", "list", "short", "sink-inputs"])
    except Exception:
        sink_inputs = ""

    sink_ids: list[str] = []
    for line in sink_inputs.splitlines():
        cols = line.split("\t")
        if len(cols) >= 2 and cols[1].strip():
            sink_ids.append(cols[1].strip())

    try:
        sinks = run(["pactl", "list", "short", "sinks"])
    except Exception:
        sinks = ""

    sink_id_to_name: dict[str, str] = {}
    for line in sinks.splitlines():
        cols = line.split("\t")
        if len(cols) >= 2:
            sink_id_to_name[cols[0].strip()] = cols[1].strip()

    if sink_ids:
        from collections import Counter

        most_common_sink_id, _count = Counter(sink_ids).most_common(1)[0]
        name = sink_id_to_name.get(most_common_sink_id, "")
        if name:
            return name

    # Next best: any sink marked RUNNING.
    for line in sinks.splitlines():
        cols = line.split("\t")
        if len(cols) >= 2 and cols[-1].strip() == "RUNNING":
            return cols[1].strip()

    # Fallback: default sink.
    try:
        return pactl_default_sink()
    except Exception:
        return ""


def resolve_pulse_source(source: str) -> str:
    override = os.environ.get("LIVE_CAPTION_PULSE_SOURCE", "").strip()
    if override:
        return override

    if source == "monitor":
        sink = pactl_active_sink()
        if not sink:
            raise RuntimeError("Couldn't detect default sink (pactl get-default-sink)")
        return f"{sink}.monitor"
    if source == "input":
        default_source = pactl_default_source()
        if not default_source:
            raise RuntimeError("Couldn't detect default source (pactl info)")
        return default_source
    raise RuntimeError(f"Invalid source: {source}")


@dataclass(frozen=True)
class Args:
    source: str
    pulse_source: str
    model: str
    language: Optional[str]
    device: str
    compute_type: str
    sample_rate: int
    window_seconds: float
    stride_seconds: float
    block_seconds: float
    beam_size: int
    best_of: int
    temperature: float
    no_speech_threshold: float
    log_prob_threshold: float
    vad_filter: bool
    parec_latency_msec: int


def parse_args(argv: list[str]) -> Args:
    parser = argparse.ArgumentParser(description="System audio -> live captions")
    parser.add_argument(
        "--source",
        choices=["monitor", "input"],
        default=os.environ.get("LIVE_CAPTION_SOURCE", "monitor"),
        help="Audio source to capture (default: monitor)",
    )
    parser.add_argument(
        "--pulse-source",
        default=os.environ.get("LIVE_CAPTION_PULSE_SOURCE", ""),
        help="Override PulseAudio source name (e.g. <sink>.monitor or @DEFAULT_MONITOR@)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("LIVE_CAPTION_MODEL", "tiny"),
        help="faster-whisper model name or local path (default: tiny)",
    )
    parser.add_argument(
        "--language",
        default=os.environ.get("LIVE_CAPTION_LANGUAGE") or None,
        help="Language code (e.g. en, bn). Auto-detect if omitted.",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("LIVE_CAPTION_DEVICE", "cpu"),
        help="Device for faster-whisper (cpu/cuda). Default: cpu",
    )
    parser.add_argument(
        "--compute-type",
        default=os.environ.get("LIVE_CAPTION_COMPUTE_TYPE", "int8"),
        help="Compute type for faster-whisper (default: int8)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=int(os.environ.get("LIVE_CAPTION_SAMPLE_RATE", "16000")),
        help="Capture sample rate (default: 16000)",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=float(os.environ.get("LIVE_CAPTION_WINDOW_SECONDS", "1.0")),
        help="Transcription window size (default: 1.0)",
    )
    parser.add_argument(
        "--stride-seconds",
        type=float,
        default=float(os.environ.get("LIVE_CAPTION_STRIDE_SECONDS", "0.5")),
        help="How often to run transcription (default: 0.5)",
    )
    parser.add_argument(
        "--block-seconds",
        type=float,
        default=float(os.environ.get("LIVE_CAPTION_BLOCK_SECONDS", "0.10")),
        help="Audio read block size (default: 0.10)",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=int(os.environ.get("LIVE_CAPTION_BEAM_SIZE", "1")),
        help="Beam size (default: 1)",
    )
    parser.add_argument(
        "--best-of",
        type=int,
        default=int(os.environ.get("LIVE_CAPTION_BEST_OF", "1")),
        help="Number of candidates when sampling (default: 1)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.environ.get("LIVE_CAPTION_TEMPERATURE", "0.0")),
        help="Decoding temperature (default: 0.0)",
    )
    parser.add_argument(
        "--no-speech-threshold",
        type=float,
        default=float(os.environ.get("LIVE_CAPTION_NO_SPEECH_THRESHOLD", "1.0")),
        help="No-speech probability threshold (default: 1.0)",
    )
    parser.add_argument(
        "--log-prob-threshold",
        type=float,
        default=float(os.environ.get("LIVE_CAPTION_LOG_PROB_THRESHOLD", "-2.0")),
        help="Log probability threshold (default: -2.0)",
    )
    parser.add_argument(
        "--vad-filter",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("LIVE_CAPTION_VAD_FILTER", "0") not in {"0", "false", "False"},
        help="Enable voice activity detection filter (default: false)",
    )
    parser.add_argument(
        "--parec-latency-msec",
        type=int,
        default=int(os.environ.get("LIVE_CAPTION_PAREC_LATENCY_MSEC", "30")),
        help="Requested parec latency in msec (default: 30)",
    )
    ns = parser.parse_args(argv)
    return Args(
        source=ns.source,
        pulse_source=(ns.pulse_source or "").strip(),
        model=ns.model,
        language=ns.language,
        device=ns.device,
        compute_type=ns.compute_type,
        sample_rate=ns.sample_rate,
        window_seconds=ns.window_seconds,
        stride_seconds=ns.stride_seconds,
        block_seconds=ns.block_seconds,
        beam_size=ns.beam_size,
        best_of=ns.best_of,
        temperature=ns.temperature,
        no_speech_threshold=ns.no_speech_threshold,
        log_prob_threshold=ns.log_prob_threshold,
        vad_filter=ns.vad_filter,
        parec_latency_msec=ns.parec_latency_msec,
    )


def load_model(model: str, device: str, compute_type: str):
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except Exception as exc:
        eprint(
            "live-caption: failed to import faster-whisper:",
            repr(exc),
        )
        eprint(
            "live-caption: install system-wide (Arch):",
            "pacman -S python-faster-whisper python-numpy",
        )
        raise
    return WhisperModel(model, device=device, compute_type=compute_type)


def pcm_s16le_to_float32(raw: bytes):
    import numpy as np  # type: ignore

    pcm = np.frombuffer(raw, dtype=np.int16)
    if pcm.size == 0:
        return pcm.astype(np.float32)
    return pcm.astype(np.float32) / 32768.0


def reader_thread(
    stop: threading.Event,
    proc: subprocess.Popen[bytes],
    out_q: "queue.Queue[object]",
    block_bytes: int,
) -> None:
    try:
        while not stop.is_set():
            data = proc.stdout.read(block_bytes)
            if not data:
                return
            try:
                out_q.put_nowait(data)
            except queue.Full:
                # Drop older audio instead of falling behind; live captions should stay "live".
                try:
                    out_q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    out_q.put_nowait(data)
                except queue.Full:
                    pass
    finally:
        stop.set()


def iter_new_segments(
    model,
    audio,
    language: Optional[str],
    beam_size: int,
    best_of: int,
    temperature: float,
    no_speech_threshold: float,
    log_prob_threshold: float,
    vad_filter: bool,
):
    segments, _info = model.transcribe(
        audio,
        language=language,
        beam_size=beam_size,
        best_of=best_of,
        temperature=temperature,
        no_speech_threshold=no_speech_threshold,
        log_prob_threshold=log_prob_threshold,
        vad_filter=vad_filter,
        condition_on_previous_text=False,
        word_timestamps=False,
    )
    return segments


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    if args.window_seconds <= 0:
        eprint("live-caption: --window-seconds must be > 0")
        return 2
    if args.stride_seconds <= 0:
        eprint("live-caption: --stride-seconds must be > 0")
        return 2
    if args.block_seconds <= 0:
        eprint("live-caption: --block-seconds must be > 0")
        return 2
    if args.best_of <= 0:
        eprint("live-caption: --best-of must be > 0")
        return 2
    if args.no_speech_threshold < 0:
        eprint("live-caption: --no-speech-threshold must be >= 0")
        return 2
    if args.parec_latency_msec < 0:
        eprint("live-caption: --parec-latency-msec must be >= 0")
        return 2
    if args.sample_rate not in {8000, 12000, 16000, 22050, 24000, 44100, 48000}:
        eprint("live-caption: unusual sample rate:", args.sample_rate)

    try:
        if args.pulse_source:
            pulse_source = args.pulse_source
        else:
            pulse_source = resolve_pulse_source(args.source)
    except Exception as exc:
        eprint(f"live-caption: {exc}")
        return 1

    if not pactl_has_source(pulse_source):
        eprint(f"live-caption: Pulse source not found: {pulse_source}")
        return 1

    try:
        eprint(f"status: source={pulse_source}")
        eprint("status: loading model…")
        model = load_model(args.model, args.device, args.compute_type)
        eprint("status: ready")
    except Exception:
        return 1

    stop = threading.Event()

    def handle_signal(_signum: int, _frame) -> None:  # type: ignore[no-untyped-def]
        stop.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    parec_cmd = [
        "parec",
        f"--device={pulse_source}",
        "--format=s16le",
        f"--rate={args.sample_rate}",
        "--channels=1",
    ]
    if args.parec_latency_msec > 0:
        parec_cmd.append(f"--latency-msec={args.parec_latency_msec}")
    try:
        proc = subprocess.Popen(
            parec_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        eprint("live-caption: parec not found")
        return 1

    if proc.stdout is None:
        eprint("live-caption: failed to read parec stdout")
        return 1

    block_samples = max(1, int(args.sample_rate * args.block_seconds))
    block_bytes = block_samples * 2
    q_max = max(4, int(args.window_seconds * args.sample_rate / block_samples) + 4)
    q: "queue.Queue[object]" = queue.Queue(maxsize=q_max)
    t = threading.Thread(target=reader_thread, args=(stop, proc, q, block_bytes), daemon=True)
    t.start()

    import numpy as np  # type: ignore

    window_samples = int(args.sample_rate * args.window_seconds)
    stride_samples = int(args.sample_rate * args.stride_seconds)

    chunks: "deque[np.ndarray]" = deque()
    chunks_len = 0
    total_samples = 0
    samples_since_transcribe = 0
    last_emitted_end_s = 0.0

    try:
        while not stop.is_set():
            try:
                raw = q.get(timeout=0.2)
            except queue.Empty:
                if proc.poll() is not None:
                    break
                continue

            if not isinstance(raw, (bytes, bytearray)):
                continue
            audio_chunk = pcm_s16le_to_float32(bytes(raw))
            if audio_chunk.size == 0:
                continue

            total_samples += int(audio_chunk.size)
            samples_since_transcribe += int(audio_chunk.size)

            chunks.append(audio_chunk)
            chunks_len += int(audio_chunk.size)

            while chunks_len > window_samples and len(chunks) > 1:
                dropped = chunks.popleft()
                chunks_len -= int(dropped.size)

            if samples_since_transcribe < stride_samples:
                continue
            samples_since_transcribe = 0

            window = np.concatenate(list(chunks)) if chunks else np.empty((0,), dtype=np.float32)
            if window.size < int(args.sample_rate * min(0.3, args.window_seconds)):
                continue

            window_start_s = (total_samples - int(window.size)) / float(args.sample_rate)

            try:
                segments = iter_new_segments(
                    model,
                    window,
                    language=args.language,
                    beam_size=args.beam_size,
                    best_of=args.best_of,
                    temperature=args.temperature,
                    no_speech_threshold=args.no_speech_threshold,
                    log_prob_threshold=args.log_prob_threshold,
                    vad_filter=args.vad_filter,
                )
            except Exception:
                continue

            for seg in segments:
                text = (getattr(seg, "text", "") or "").strip()
                if not text:
                    continue

                end_s = float(getattr(seg, "end", 0.0) or 0.0)
                end_abs_s = window_start_s + end_s
                if end_abs_s <= (last_emitted_end_s + 0.05):
                    continue

                print(text, flush=True)
                last_emitted_end_s = end_abs_s

    finally:
        stop.set()
        try:
            proc.terminate()
        except Exception:
            pass
        try:
            proc.wait(timeout=1)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
