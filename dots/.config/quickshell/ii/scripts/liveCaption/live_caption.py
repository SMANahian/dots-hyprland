#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import select
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Debug logging
DEBUG_FILE = "/tmp/live_caption_debug.log"

def log_debug(msg: str):
    try:
        with open(DEBUG_FILE, "a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    except Exception:
        pass

def _find_first(model_dir: Path, patterns: list[str]) -> Path | None:
    for pattern in patterns:
        matches = sorted(model_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def _find_onnx_component(model_dir: Path, name: str) -> Path | None:
    return _find_first(
        model_dir,
        [
            f"{name}.int8.onnx",
            f"{name}*.int8.onnx",
            f"{name}*int8*.onnx",
            f"{name}.onnx",
            f"{name}*.onnx",
            f"*{name}*.onnx",
        ],
    )


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return (
        0x3400 <= code <= 0x4DBF  # CJK Unified Ideographs Extension A
        or 0x4E00 <= code <= 0x9FFF  # CJK Unified Ideographs
        or 0x3040 <= code <= 0x30FF  # Hiragana + Katakana
    )


def _tokens_look_like_bpe(tokens_path: Path) -> bool:
    try:
        with tokens_path.open("r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                if i > 5000:
                    break
                line = line.strip()
                if not line:
                    continue
                token = line.split(maxsplit=1)[0]
                if token != "▁" and "▁" in token:
                    return True
                if token.startswith("##"):
                    return True
    except Exception:
        return False
    return False


def _tokens_contain_cjk(tokens_path: Path) -> bool:
    try:
        with tokens_path.open("r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                if i > 5000:
                    break
                line = line.strip()
                if not line:
                    continue
                token = line.split(maxsplit=1)[0]
                if any(_is_cjk(ch) for ch in token):
                    return True
    except Exception:
        return False
    return False


def _trim_history(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _looks_all_caps_sentence(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 8:
        return False
    upper = sum(1 for c in letters if c.isupper())
    lower = sum(1 for c in letters if c.islower())
    if lower > 0:
        return False
    return upper / len(letters) >= 0.95 and (" " in text or len(letters) >= 12)


def _sanitize_tokens(tokens_path: Path) -> Path:
    """
    Reads the tokens file and writes a sanitized version (skipping empty lines)
    to a temporary file. Returns the path to the temporary file.
    """
    try:
        with tokens_path.open("r", encoding="utf-8", errors="ignore") as f:
            lines = [line for line in f if line.strip()]
        
        import tempfile
        fd, path = tempfile.mkstemp(suffix="_tokens.txt", text=True)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.writelines(lines)
        log_debug(f"Sanitized tokens file created at {path}")
        return Path(path)
    except Exception as e:
        log_debug(f"Failed to sanitize tokens: {e}")
        return tokens_path


def _sentence_case(text: str) -> str:
    lowered = text.lower()
    for i, ch in enumerate(lowered):
        if ch.isalpha():
            return lowered[:i] + ch.upper() + lowered[i + 1 :]
    return lowered


def _normalize_caption(text: str) -> str:
    trimmed = (text or "").strip()
    if not trimmed:
        return ""

    if _looks_all_caps_sentence(trimmed):
        return _sentence_case(trimmed)
    return trimmed


def _run_text(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True).strip()


def _default_pulse_source(source_kind: str) -> str:
    if source_kind == "monitor":
        sink = _run_text(["pactl", "get-default-sink"])
        if not sink:
            raise RuntimeError("pactl returned an empty default sink")
        return f"{sink}.monitor"

    if source_kind == "input":
        info = _run_text(["pactl", "info"])
        for line in info.splitlines():
            if line.startswith("Default Source:"):
                value = line.split(":", 1)[1].strip()
                if not value:
                    break
                return value
        raise RuntimeError("failed to find default source in `pactl info` output")

    raise ValueError(f"Unknown source kind: {source_kind}")


def _pulse_source_exists(source: str) -> bool:
    try:
        sources = _run_text(["pactl", "list", "short", "sources"])
    except Exception:
        return False
    for line in sources.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        name = parts[1].strip()
        if name == source:
            return True
    return False


class _StopFlag:
    def __init__(self) -> None:
        self.value = False

    def set(self, *_args) -> None:
        self.value = True


def _start_audio_proc(target: str, sample_rate: int) -> tuple[subprocess.Popen, str]:
    """
    Start an audio capture process for the given PulseAudio/PipeWire source.
    
    Uses ffmpeg as primary method (most reliable for monitor sources),
    falls back to parec, then pw-record.
    """
    
    backends = [
        ("ffmpeg", [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "pulse", "-i", target,
            "-ac", "1", "-ar", str(sample_rate),
            "-f", "s16le", "-acodec", "pcm_s16le", "pipe:1"
        ]),
        ("parec", [
            "parec", f"--device={target}",
            "--format=s16le", f"--rate={sample_rate}", "--channels=1"
        ]),
        ("pw-record", [
            "pw-record", "--target", target,
            "--rate", str(sample_rate), "--channels", "1",
            "--format", "s16", "--raw", "-"
        ])
    ]

    for name, cmd in backends:
        if not shutil.which(cmd[0]):
            continue
            
        log_debug(f"Starting {name} audio proc: {' '.join(cmd)}")
        
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                start_new_session=True,
            )
            
            time.sleep(0.15)  # Give it a moment to fail
            
            if proc.poll() is None:
                if proc.stdout is not None:
                    return proc, name
            else:
                err = (proc.stderr.read() if proc.stderr else b"").decode(errors="replace").strip()
                log_debug(f"{name} failed: {err}")
                try:
                    proc.wait(timeout=0.2)
                except Exception:
                    pass
        except Exception as e:
            log_debug(f"{name} exception: {e}")

    raise RuntimeError(f"All audio backends failed for target {target}")


def main() -> int:
    # Clear debug log
    try:
        with open(DEBUG_FILE, "w") as f:
            f.write("Starting live caption...\n")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Live caption from system audio using sherpa-onnx (streaming).")

    parser.add_argument("--model-dir", default="", help="Directory containing sherpa-onnx streaming model files.")
    parser.add_argument("--source", choices=["monitor", "input", "both"], default="monitor")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--provider", default="cpu", help="onnxruntime provider: cpu/cuda/tensorrt/...")
    parser.add_argument(
        "--decoding-method",
        default="modified_beam_search",
        choices=["greedy_search", "modified_beam_search"],
        help="Decoding method. modified_beam_search is usually more accurate (slightly slower).",
    )
    parser.add_argument(
        "--max-active-paths",
        type=int,
        default=4,
        help="Beam width for modified_beam_search (higher = more accurate, slower).",
    )

    parser.add_argument("--encoder", default="", help="Override encoder ONNX path.")
    parser.add_argument("--decoder", default="", help="Override decoder ONNX path.")
    parser.add_argument("--joiner", default="", help="Override joiner ONNX path.")
    parser.add_argument("--tokens", default="", help="Override tokens.txt path.")
    parser.add_argument("--model-type", default="", help="Optional sherpa-onnx model_type for transducer.")
    parser.add_argument(
        "--modeling-unit",
        default="",
        help="Optional modeling unit (e.g. bpe/char/cjkchar). If unset, inferred when possible.",
    )
    parser.add_argument("--bpe-vocab", default="", help="Optional BPE vocab/model path (when modeling-unit=bpe).")

    parser.add_argument("--enable-endpoint", action="store_true", help="Enable endpoint detection.")
    parser.add_argument("--rule1-min-trailing-silence", type=float, default=2.4)
    parser.add_argument("--rule2-min-trailing-silence", type=float, default=1.2)
    parser.add_argument("--rule3-min-utterance-length", type=float, default=20.0)

    parser.add_argument("--update-interval-ms", type=int, default=250, help="Throttle stdout updates.")
    parser.add_argument("--history-chars", type=int, default=240, help="Keep up to N chars of caption history.")
    parser.add_argument("--no-history", action="store_true", help="Do not accumulate history in the backend.")

    args = parser.parse_args()
    
    log_debug(f"Args: source={args.source}, model_dir={args.model_dir}")

    model_dir_str = args.model_dir.strip()
    if not model_dir_str:
        print("error: --model-dir is required", file=sys.stderr)
        return 2

    model_dir = Path(os.path.expanduser(model_dir_str)).resolve()
    if not model_dir.exists():
        print(f"error: model dir does not exist: {model_dir}", file=sys.stderr)
        return 2

    encoder = Path(os.path.expanduser(args.encoder)).resolve() if args.encoder else _find_onnx_component(model_dir, "encoder")
    decoder = Path(os.path.expanduser(args.decoder)).resolve() if args.decoder else _find_onnx_component(model_dir, "decoder")
    joiner = Path(os.path.expanduser(args.joiner)).resolve() if args.joiner else _find_onnx_component(model_dir, "joiner")
    tokens = Path(os.path.expanduser(args.tokens)).resolve() if args.tokens else _find_first(model_dir, ["tokens.txt"])

    if not encoder or not decoder or not joiner or not tokens:
        print(
            "error: missing model files. Need encoder/decoder/joiner .onnx and tokens.txt under model dir.",
            file=sys.stderr,
        )
        return 2

    # Sanitize tokens file to avoid "given :" errors from empty lines
    tokens = _sanitize_tokens(tokens)

    bpe_vocab = Path(os.path.expanduser(args.bpe_vocab)).resolve() if args.bpe_vocab else _find_first(
        model_dir, ["bpe.vocab", "bpe.model", "*bpe*.vocab", "*bpe*.model"]
    )
    
    log_debug(f"Model files: tokens={tokens}, bpe_vocab={bpe_vocab}")
    
    modeling_unit = args.modeling_unit.strip()
    if not modeling_unit:
        if bpe_vocab or _tokens_look_like_bpe(tokens):
            modeling_unit = "bpe"
        elif _tokens_contain_cjk(tokens):
            modeling_unit = "cjkchar"
        else:
            modeling_unit = "char"

    try:
        import sherpa_onnx  # type: ignore
    except Exception as exc:
        print(
            f"error: sherpa_onnx import failed: {exc} (install 'sherpa-onnx' into your Quickshell venv; Settings → Services → Live caption → Install python deps)",
            file=sys.stderr,
        )
        return 3
    try:
        import numpy as np  # type: ignore
    except Exception as exc:
        print(
            f"error: numpy import failed: {exc} (install python deps; Settings → Services → Live caption → Install python deps)",
            file=sys.stderr,
        )
        return 3

    # sherpa-onnx expects a text file (token score) for bpe_vocab argument.
    # If we found a binary .model file (SentencePiece), we shouldn't pass it as bpe_vocab
    # because it causes a crash when parsing.
    pass_bpe_vocab = ""
    if bpe_vocab:
        # Check if it looks like a binary model file
        if bpe_vocab.suffix == ".model":
            log_debug(f"Skipping bpe_vocab argument because {bpe_vocab.name} looks like a binary model")
            pass_bpe_vocab = ""
        else:
            pass_bpe_vocab = str(bpe_vocab)

    recognizer_kwargs = dict(
        tokens=str(tokens),
        encoder=str(encoder),
        decoder=str(decoder),
        joiner=str(joiner),
        num_threads=args.num_threads,
        sample_rate=args.sample_rate,
        feature_dim=80,
        enable_endpoint_detection=args.enable_endpoint,
        rule1_min_trailing_silence=args.rule1_min_trailing_silence,
        rule2_min_trailing_silence=args.rule2_min_trailing_silence,
        rule3_min_utterance_length=args.rule3_min_utterance_length,
        decoding_method=args.decoding_method,
        provider=args.provider,
        model_type=args.model_type,
        modeling_unit=modeling_unit,
        bpe_vocab=pass_bpe_vocab,
    )
    if args.decoding_method != "greedy_search":
        recognizer_kwargs["max_active_paths"] = max(1, args.max_active_paths)
    try:
        recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(**recognizer_kwargs)
    except TypeError:
        recognizer_kwargs.pop("max_active_paths", None)
        recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(**recognizer_kwargs)
    stream = recognizer.create_stream()

    try:
        targets = []
        if args.source == "both":
            targets.append(_default_pulse_source("monitor"))
            targets.append(_default_pulse_source("input"))
        else:
            targets.append(_default_pulse_source(args.source))
    except Exception as exc:
        print(f"error: failed to resolve audio source: {exc}", file=sys.stderr)
        return 4

    for t in targets:
        log_debug(f"Resolved target: {t}")
        if not _pulse_source_exists(t):
            print(
                f"error: audio source not found: {t} (check `pactl list short sources`)",
                file=sys.stderr,
            )
            return 4

    stop = _StopFlag()
    signal.signal(signal.SIGTERM, stop.set)
    signal.signal(signal.SIGINT, stop.set)

    streams = []
    try:
        for t in targets:
            proc, backend = _start_audio_proc(t, args.sample_rate)
            fd = proc.stdout.fileno()
            try:
                os.set_blocking(fd, False)
            except AttributeError:
                pass
            streams.append({
                "proc": proc,
                "backend": backend,
                "fd": fd,
                "buffer": bytearray(),
                "target": t
            })
    except Exception as e:
        print(f"error: failed to start audio recording: {e}", file=sys.stderr)
        log_debug(f"Audio start failed: {e}")
        for s in streams:
            s["proc"].terminate()
        return 4

    bytes_per_sample = 2
    chunk_ms = 60
    chunk_bytes = int(args.sample_rate * (chunk_ms / 1000.0) * bytes_per_sample)

    history_text = ""
    last_emitted = None
    last_emit_ts = 0.0
    update_interval_s = max(args.update_interval_ms, 0) / 1000.0

    exit_code = 0

    log_debug("Starting loop")

    try:
        while not stop.value:
            fds = [s["fd"] for s in streams]
            readable, _, _ = select.select(fds, [], [], 0.1)
            if stop.value:
                break
            
            for s in streams:
                if s["fd"] in readable:
                    try:
                        chunk = os.read(s["fd"], chunk_bytes)
                        if chunk:
                            s["buffer"].extend(chunk)
                        else:
                            # EOF
                            log_debug(f"EOF on stream {s['target']}")
                            stop.set()
                    except BlockingIOError:
                        pass
            
            # Process if all streams have enough data
            while all(len(s["buffer"]) >= chunk_bytes for s in streams):
                mixed_samples = None
                for s in streams:
                    data = bytes(s["buffer"][:chunk_bytes])
                    del s["buffer"][:chunk_bytes]
                    
                    # Ensure even length
                    if len(data) % 2 != 0:
                        data = data[:-1]
                    
                    samples = (np.frombuffer(data, dtype=np.int16).astype(np.float32)) / 32768.0
                    if mixed_samples is None:
                        mixed_samples = samples
                    else:
                        mixed_samples += samples
                
                if mixed_samples is not None:
                    # Average if mixing
                    if len(streams) > 1:
                        mixed_samples /= len(streams)
                    
                    stream.accept_waveform(float(args.sample_rate), mixed_samples)

                    while recognizer.is_ready(stream):
                        recognizer.decode_stream(stream)

                    current = _normalize_caption(recognizer.get_result(stream))
                    if args.no_history:
                        display = current
                    else:
                        display = (history_text + " " + current).strip() if history_text else current
                        if args.history_chars > 0:
                            display = _trim_history(display, args.history_chars)

                    now = time.monotonic()
                    if display != last_emitted and (update_interval_s == 0 or now - last_emit_ts >= update_interval_s):
                        print(display, flush=True)
                        last_emitted = display
                        last_emit_ts = now

                    if args.enable_endpoint and recognizer.is_endpoint(stream):
                        if current and not args.no_history:
                            history_text = (history_text + " " + current).strip() if history_text else current
                            history_text = _trim_history(history_text, args.history_chars)

                        recognizer.reset(stream)

                        if not args.no_history:
                            if history_text != last_emitted:
                                print(history_text, flush=True)
                                last_emitted = history_text
                                last_emit_ts = time.monotonic()

        if not stop.value:
            for s in streams:
                rc = s["proc"].poll()
                if rc not in (None, 0):
                    err = (s["proc"].stderr.read() if s["proc"].stderr else b"").decode(errors="replace").strip()
                    msg = f"error: audio backend {s['backend']} exited with code {rc} for target {s['target']}"
                    if err:
                        msg += f": {err}"
                    print(msg, file=sys.stderr)
                    exit_code = 4

    finally:
        log_debug("Exiting")
        for s in streams:
            try:
                s["proc"].terminate()
            except Exception:
                pass

        for s in streams:
            try:
                s["proc"].wait(timeout=2)
            except Exception:
                try:
                    s["proc"].kill()
                except Exception:
                    pass

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
