"""Feature extraction matching the existing AV-ASD baseline stages."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import json
import cv2
import numpy as np
import pandas as pd
import librosa
import soundfile as sf

FFMPEG = shutil.which("ffmpeg")

def extract_visual_features(video_id):
    """Extract basic appearance and temporal-change features."""
    clip_path = CLIP_DIR / f'{video_id}.mp4'
    cap = cv2.VideoCapture(str(clip_path))
    frame_values = []
    changes = []
    previous_gray = None
    try:
        if not cap.isOpened():
            raise RuntimeError('Cannot open the video.')
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if not np.isfinite([fps, frame_count]).all():
            raise ValueError('Invalid video metadata.')
        if fps <= 0 or frame_count <= 0:
            raise ValueError('Invalid FPS or frame count.')
        duration = frame_count / fps
        targets = np.unique(np.rint(np.arange(0, duration, 1.0) * fps).astype(int))
        targets = targets[targets < int(frame_count)]
        for index, frame_number in enumerate(targets):
            if not cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number)):
                raise RuntimeError('Frame seeking is unsupported.')
            success, frame = cap.read()
            if not success or frame is None:
                raise RuntimeError(f'Cannot decode frame {frame_number}.')
            height, width = frame.shape[:2]
            scale = 224 / max(height, width)
            resized = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            gray_float = gray.astype(np.float32) / 255.0
            edges = cv2.Canny(gray, 100, 200)
            rgb_mean = rgb.mean(axis=(0, 1)) / 255.0
            frame_values.append([*rgb_mean, float(gray_float.mean()), float(gray_float.std()), float((edges > 0).mean())])
            if previous_gray is not None:
                changes.append(float(np.abs(gray_float - previous_gray).mean()))
            previous_gray = gray_float
        if len(frame_values) < 2:
            raise ValueError('At least two sampled frames are required.')
        values = np.asarray(frame_values)
        names = ['red', 'green', 'blue', 'brightness', 'contrast', 'edge_density']
        features = {}
        for column, name in enumerate(names):
            features[f'visual_{name}_mean'] = float(values[:, column].mean())
            features[f'visual_{name}_std'] = float(values[:, column].std())
        features['visual_temporal_change_mean'] = float(np.mean(changes))
        features['visual_temporal_change_std'] = float(np.std(changes))
        if not np.isfinite(list(features.values())).all():
            raise ValueError('Invalid feature values.')
        return (features, len(frame_values))
    finally:
        cap.release()

def retry_visual_features(video_id):
    """Decode sequentially with FFmpeg and handle short clips."""
    path = CLIP_DIR / f'{video_id}.mp4'
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError('Cannot open the clip.')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    finally:
        cap.release()
    if not np.isfinite([fps, width, height]).all():
        raise ValueError('Invalid video metadata.')
    if min(fps, width, height) <= 0:
        raise ValueError('Invalid FPS or dimensions.')
    scale = 224 / max(width, height)
    out_width = max(1, round(width * scale))
    out_height = max(1, round(height * scale))
    result = subprocess.run([FFMPEG, '-nostdin', '-v', 'error', '-xerror', '-threads', '1', '-i', str(path), '-map', '0:v:0', '-an', '-vf', f'scale={out_width}:{out_height}:flags=area', '-fps_mode', 'passthrough', '-pix_fmt', 'bgr24', '-c:v', 'rawvideo', '-threads', '1', '-f', 'rawvideo', 'pipe:1'], capture_output=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode(errors='replace').strip())
    frame_bytes = out_width * out_height * 3
    if not result.stdout or len(result.stdout) % frame_bytes != 0:
        raise ValueError('Empty or incomplete decoded frame data.')
    frames = np.frombuffer(result.stdout, dtype=np.uint8).reshape(-1, out_height, out_width, 3)
    actual_count = len(frames)
    if actual_count < 2:
        raise ValueError('The clip contains fewer than two decoded frames.')
    targets = np.unique(np.rint(np.arange(0, actual_count / fps, 1.0) * fps).astype(int))
    targets = targets[targets < actual_count]
    short_fallback = len(targets) < 2
    if short_fallback:
        targets = np.array([0, actual_count - 1])
    frame_values = []
    changes = []
    previous_gray = None
    for frame_number in targets:
        frame = frames[frame_number]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_float = gray.astype(np.float32) / 255.0
        edges = cv2.Canny(gray, 100, 200)
        frame_values.append([*rgb.mean(axis=(0, 1)) / 255.0, float(gray_float.mean()), float(gray_float.std()), float((edges > 0).mean())])
        if previous_gray is not None:
            changes.append(float(np.abs(gray_float - previous_gray).mean()))
        previous_gray = gray_float
    values = np.asarray(frame_values)
    names = ['red', 'green', 'blue', 'brightness', 'contrast', 'edge_density']
    features = {}
    for column, name in enumerate(names):
        features[f'visual_{name}_mean'] = float(values[:, column].mean())
        features[f'visual_{name}_std'] = float(values[:, column].std())
    features['visual_temporal_change_mean'] = float(np.mean(changes))
    features['visual_temporal_change_std'] = float(np.std(changes))
    if not np.isfinite(list(features.values())).all():
        raise ValueError('Invalid feature values.')
    return (features, {'sampled_frames': len(targets), 'decoded_frames': actual_count, 'short_clip_fallback': short_fallback, 'mean_sample_interval_seconds': float(np.diff(targets).mean() / fps)})

def extract_audio(path, temp_path):
    if not FFMPEG:
        raise RuntimeError("FFmpeg was not found on PATH.")
    result = subprocess.run([FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "error",
        "-y", "-threads", "1", "-i", str(path), "-map", "0:a:0", "-vn",
        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(temp_path)],
        capture_output=True, text=True, timeout=300)
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    y, sr = sf.read(temp_path, dtype='float32')
    if sr != 16000 or y.ndim != 1 or len(y) < 512:
        raise ValueError('Unexpected sample rate, channels, or audio length.')
    if not np.isfinite(y).all():
        raise ValueError('Audio contains invalid values.')
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=512, hop_length=160, n_mels=40)
    rms = librosa.feature.rms(y=y, frame_length=512, hop_length=160)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=512, hop_length=160)
    values = {}
    for index in range(20):
        values[f'mfcc_{index + 1}_mean'] = float(mfcc[index].mean())
        values[f'mfcc_{index + 1}_std'] = float(mfcc[index].std())
    for name, array in [('rms', rms), ('zcr', zcr)]:
        values[f'{name}_mean'] = float(array.mean())
        values[f'{name}_std'] = float(array.std())
    if not np.isfinite(list(values.values())).all():
        raise ValueError('Features contain invalid values.')
    return values, bool(np.max(np.abs(y)) == 0)


def extract_video(path):
    """Process one short local video without modifying project datasets."""
    path = Path(path).resolve()
    if not path.is_file() or path.stat().st_size > 200 * 1024**2:
        raise ValueError("Provide an existing video smaller than 200 MiB.")
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if not cap.isOpened() or not np.isfinite([fps, count]).all() or min(fps, count) <= 0:
            raise ValueError("Invalid video metadata.")
        duration = count / fps
        if duration > 120 or count > 7200:
            raise ValueError("Pilot limit: 120 seconds and 7200 frames.")
    finally:
        cap.release()
    # The original visual functions expect a clip directory and stem.
    global CLIP_DIR
    CLIP_DIR = path.parent
    if path.suffix.lower() != ".mp4":
        raise ValueError("This pilot accepts MP4 files only.")
    cv2.setNumThreads(1)
    try:
        visual, sampled = extract_visual_features(path.stem)
        decoder = "OpenCV"
    except (ValueError, RuntimeError):
        visual, details = retry_visual_features(path.stem)
        sampled = details["sampled_frames"]
        decoder = "FFmpeg fallback"
    with tempfile.TemporaryDirectory(prefix="av_asd_audio_", dir=path.parent) as folder:
        audio, silent = extract_audio(path, Path(folder) / "audio.wav")
    values = {**audio, **visual}
    if len(values) != 58 or not np.isfinite(list(values.values())).all():
        raise ValueError("Expected 58 finite features.")
    return values, {"duration_seconds": duration, "sampled_frames": sampled,
        "decoder": decoder, "all_zero_audio": silent}
