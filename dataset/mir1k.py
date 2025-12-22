import os
import argparse
import pandas as pd
from tqdm import tqdm
import librosa
import soundfile as sf
import numpy as np
from scipy.interpolate import interp1d


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--wav_dir", type=str, required=True)
    parser.add_argument("--pitch_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--sr", type=int, default=16000)
    return parser.parse_args()


def midi_to_hz(midi_note):
    if midi_note <= 0:
        return 0.0
    return 440.0 * (2.0 ** ((midi_note - 69.0) / 12.0))


def process_dataset(args):
    df_info = pd.read_csv(args.csv_path)

    for _, row in tqdm(df_info.iterrows(), total=df_info.shape[0]):
        filename = str(row.iloc[0])
        split = str(row.iloc[2])

        wav_path_in = os.path.join(args.wav_dir, filename)
        split_out_dir = os.path.join(args.out_dir, split)
        os.makedirs(split_out_dir, exist_ok=True)

        try:
            audio, _ = librosa.load(wav_path_in, sr=args.sr, mono=False)

            pv_filename = filename.replace(".wav", ".pv")
            pv_in_path = os.path.join(args.pitch_dir, pv_filename)

            if os.path.exists(pv_in_path):
                f0_midi = np.loadtxt(pv_in_path)

                old_times = 0.020 + np.arange(len(f0_midi)) * 0.02
                new_times = np.arange(0.020, old_times[-1] + 0.01, 0.01)

                f_pitch = interp1d(
                    old_times,
                    f0_midi,
                    kind="linear",
                    fill_value="extrapolate",
                    bounds_error=False,
                )
                f0_interp = f_pitch(new_times)

                is_voiced = (f0_midi > 0).astype(float)
                f_mask = interp1d(
                    old_times,
                    is_voiced,
                    kind="nearest",
                    fill_value=0,
                    bounds_error=False,
                )
                mask_interp = f_mask(new_times)
                f0_interp[mask_interp == 0] = 0.0
                f0_interp = np.nan_to_num(f0_interp, nan=0.0)

                f0_hz = np.array([midi_to_hz(midi) for midi in f0_interp])

                sf.write(
                    os.path.join(split_out_dir, filename),
                    audio.T,
                    args.sr,
                    "PCM_24",
                )

                np.savetxt(
                    os.path.join(split_out_dir, pv_filename),
                    f0_hz,
                    fmt="%.9f",
                )

        except Exception as e:
            print(f"Error processing {filename}: {e}")


if __name__ == "__main__":
    args = parse_args()
    process_dataset(args)
