import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm
import argparse
from scipy.interpolate import interp1d


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--sr", type=int, default=16000)
    return parser.parse_args()


def process_f0(csv_path, duration):
    df = pd.read_csv(csv_path, header=None, names=["time", "hz"])

    hop_sec = 0.01
    new_times = np.arange(0, duration, hop_sec)

    f_pitch = interp1d(
        df["time"],
        df["hz"],
        kind="linear",
        fill_value="extrapolate",
        bounds_error=False,
    )
    f0_interp = f_pitch(new_times)

    is_voiced = (df["hz"] > 0).astype(float)
    f_mask = interp1d(
        df["time"], is_voiced, kind="nearest", fill_value=0, bounds_error=False
    )
    mask_interp = f_mask(new_times)

    f0_interp[mask_interp == 0] = 0.0
    f0_interp = np.nan_to_num(f0_interp, nan=0.0)
    f0_interp[f0_interp < 0] = 0.0

    return f0_interp


def process_dataset():
    args = parse_args()

    audio_dir = os.path.join(args.wav_dir, "audio_stems")
    anno_dir = os.path.join(args.wav_dir, "annotation_stems")
    all_files = sorted(
        [
            f
            for f in os.listdir(audio_dir)
            if f.endswith(".wav") and not f.startswith("._")
        ]
    )
    total_files = len(all_files)

    if total_files == 0:
        return

    split_idx = int(total_files * 0.8)
    train_files = all_files[:split_idx]
    test_files = all_files[split_idx:]

    splits = {"train": train_files, "test": test_files}

    for split_name, files in splits.items():
        split_out_dir = os.path.join(args.out_dir, split_name)
        os.makedirs(split_out_dir, exist_ok=True)

        for filename in tqdm(files, desc=f"Split {split_name}"):
            base_name = os.path.splitext(filename)[0]
            wav_path = os.path.join(audio_dir, filename)
            csv_path = os.path.join(anno_dir, base_name + ".csv")

            if not os.path.exists(csv_path):
                continue

            try:
                audio, _ = librosa.load(wav_path, sr=args.sr)
                duration = len(audio) / args.sr

                f0_resampled = process_f0(csv_path, duration)

                out_wav_path = os.path.join(split_out_dir, f"{base_name}_p.wav")
                out_pv_path = os.path.join(split_out_dir, f"{base_name}_p.pv")

                sf.write(out_wav_path, audio, args.sr, subtype="PCM_24")
                np.savetxt(out_pv_path, f0_resampled, fmt="%.6f")

            except Exception as e:
                print(f"Error {filename}: {e}")


if __name__ == "__main__":
    process_dataset()
