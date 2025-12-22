import os
import argparse
import pandas as pd
from tqdm import tqdm
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d

TRAIN_SINGERS = [
    "f1",
    "f3",
    "f4",
    "f5",
    "f6",
    "f7",
    "f9",
    "m1",
    "m2",
    "m4",
    "m6",
    "m7",
    "m8",
    "m9",
    "m11",
]

MIN_FRAMES = 129


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_dir", type=str, required=True)
    parser.add_argument("--csv_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--sr", type=int, default=16000)
    return parser.parse_args()


def get_split(filename_base):
    singer_name = filename_base.split("_")[0]
    return "train" if singer_name in TRAIN_SINGERS else "test"


def process_f0_vocalset(time_sec, f0_values, duration):
    hop_sec = 0.01
    n_frames = int(np.ceil(duration / hop_sec))
    new_times = (np.arange(n_frames) + 0.5) * hop_sec

    f_pitch = interp1d(
        time_sec, f0_values, kind="linear", fill_value="extrapolate", bounds_error=False
    )
    f0_interp = f_pitch(new_times)

    is_voiced = (f0_values > 0).astype(float)
    f_mask = interp1d(
        time_sec, is_voiced, kind="nearest", fill_value=0, bounds_error=False
    )
    mask_interp = f_mask(new_times)

    f0_interp[mask_interp == 0] = 0.0
    f0_interp = np.nan_to_num(f0_interp, nan=0.0)
    f0_interp[f0_interp < 0] = 0.0

    return f0_interp


def process_dataset(args):
    wav_root = Path(args.wav_dir)
    csv_root = Path(args.csv_dir)
    out_root = Path(args.out_dir)

    all_wav_paths = list(wav_root.rglob("*.wav"))

    for wav_path in tqdm(all_wav_paths):
        filename_base = wav_path.stem
        technique_folder = wav_path.relative_to(wav_root).parent
        split = get_split(filename_base)

        csv_path = csv_root / technique_folder / (filename_base + ".csv")

        if not csv_path.exists():
            if "slow_" in filename_base:
                fixed_name = filename_base.replace("slow_", "sow_")
                csv_path_corrected = csv_root / technique_folder / (fixed_name + ".csv")
                if csv_path_corrected.exists():
                    csv_path = csv_path_corrected
                else:
                    continue
            else:
                continue

        try:
            audio, _ = librosa.load(str(wav_path), sr=args.sr, mono=False)
            duration = (audio.shape[1] if audio.ndim > 1 else len(audio)) / args.sr

            df_frame = pd.read_csv(csv_path, skipinitialspace=True)
            df_frame.columns = [col.strip() for col in df_frame.columns]

            f0_frames = process_f0_vocalset(
                df_frame["Time (second)"].values, df_frame["F0"].values, duration
            )

            current_frames = len(f0_frames)
            if current_frames < MIN_FRAMES:
                repeats = int(np.ceil(MIN_FRAMES / current_frames))
                f0_frames = np.tile(f0_frames, repeats)
                if audio.ndim > 1:
                    audio = np.tile(audio, (1, repeats))
                else:
                    audio = np.tile(audio, repeats)

            out_split_dir = out_root / split
            out_split_dir.mkdir(parents=True, exist_ok=True)

            np.savetxt(out_split_dir / (filename_base + ".pv"), f0_frames, fmt="%.9f")
            sf.write(
                out_split_dir / (filename_base + ".wav"), audio.T, args.sr, "PCM_24"
            )

        except Exception as e:
            print(f"Error processing {filename_base}: {e}")


if __name__ == "__main__":
    args = parse_args()
    process_dataset(args)
