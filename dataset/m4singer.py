import argparse
import math
import os
import warnings

import librosa
import numpy as np
import pandas as pd
import parselmouth
import soundfile as sf
from scipy.interpolate import interp1d
from tqdm import tqdm

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--in_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--f0_min", type=float, default=80.0)
    parser.add_argument("--f0_max", type=float, default=750.0)
    return parser.parse_args()


def extract_f0_interpolated(audio_np, sr, hop_length, f0_min, f0_max):
    time_step = hop_length / sr

    snd = parselmouth.Sound(audio_np, sampling_frequency=sr)
    pitch = snd.to_pitch_ac(
        time_step=time_step,
        voicing_threshold=0.6,
        pitch_floor=f0_min,
        pitch_ceiling=f0_max,
    )

    pitch_values = pitch.selected_array["frequency"]
    pitch_times = pitch.xs()

    mel_len = int(math.ceil(len(audio_np) / hop_length))
    target_times = librosa.frames_to_time(
        np.arange(mel_len), sr=sr, hop_length=hop_length
    )

    f_pitch = interp1d(
        pitch_times, pitch_values, kind="linear", bounds_error=False, fill_value=0.0
    )
    f0_interp = f_pitch(target_times)

    is_voiced = (pitch_values > 0).astype(float)
    f_mask = interp1d(
        pitch_times, is_voiced, kind="nearest", bounds_error=False, fill_value=0.0
    )
    mask_interp = f_mask(target_times)

    f0_interp[mask_interp == 0] = 0.0
    f0_interp = np.nan_to_num(f0_interp, nan=0.0)
    f0_interp[f0_interp < 0] = 0.0

    return f0_interp


def process_dataset(args):
    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"CSV file not found: {args.csv_path}")

    df_info = pd.read_csv(args.csv_path)

    for _, row in tqdm(df_info.iterrows(), total=len(df_info)):
        relative_path = row["name"]
        split = row["split"]

        folder_name = relative_path.split("/")[0]
        file_index = os.path.splitext(relative_path.split("/")[1])[0]
        base_filename = f"{folder_name}_{file_index}"

        wav_path_in = os.path.join(args.in_dir, relative_path)
        split_out_dir = os.path.join(args.out_dir, split)
        os.makedirs(split_out_dir, exist_ok=True)

        try:
            audio, _ = librosa.load(wav_path_in, sr=args.sr, mono=False)

            sf.write(
                os.path.join(split_out_dir, f"{base_filename}.wav"),
                audio.T,
                args.sr,
                "PCM_24",
            )

            if audio.ndim > 1:
                audio_mono = np.mean(audio, axis=0)
            else:
                audio_mono = audio

            f0_sequence = extract_f0_interpolated(
                audio_mono, args.sr, 160, args.f0_min, args.f0_max
            )

            np.savetxt(
                os.path.join(split_out_dir, f"{base_filename}.pv"),
                f0_sequence,
                fmt="%.9f",
            )

        except Exception as e:
            print(f"Error processing {relative_path}: {e}")


if __name__ == "__main__":
    args = parse_args()
    process_dataset(args)
