import os
import argparse
import pandas as pd
from tqdm import tqdm
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path

TRAIN_SINGERS = ['f1', 'f3', 'f4', 'f5', 'f6', 'f7', 'f9', 'm1', 'm2', 'm4', 'm6', 'm7', 'm8', 'm9', 'm11']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav_dir', type=str, required=True)
    parser.add_argument('--csv_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--sr', type=int, default=16000)
    return parser.parse_args()

def get_split(filename_base):
    singer_name = filename_base.split('_')[0]
    return 'train' if singer_name in TRAIN_SINGERS else 'test'

def resample_f0_to_hop(time_sec, f0_values, duration, hop_ms):
    hop_sec = hop_ms / 1000.0
    n_frames_target = int(np.ceil(duration / hop_sec))
    times_target = (np.arange(n_frames_target) + 0.5) * hop_sec
    
    if times_target[-1] > duration:
        times_target = times_target[times_target < duration + hop_sec]

    voiced_mask = f0_values > 0.0
    if voiced_mask.sum() == 0:
        return np.zeros(len(times_target), dtype=np.float32)

    time_voiced = time_sec[voiced_mask]
    f0_voiced = f0_values[voiced_mask]
    
    f0_resampled = np.interp(times_target, time_voiced, f0_voiced, left=0.0, right=0.0)
    f0_resampled[f0_resampled < 0] = 0.0
    
    return f0_resampled.astype(np.float32)

def process_dataset(args):
    wav_root = Path(args.wav_dir)
    csv_root = Path(args.csv_dir)
    out_root = Path(args.out_dir)

    all_wav_paths = list(wav_root.rglob('*.wav'))

    for wav_path in tqdm(all_wav_paths):
        filename_base = wav_path.stem
        technique_folder = wav_path.relative_to(wav_root).parent
        split = get_split(filename_base)
        
        csv_path = csv_root / technique_folder / (filename_base + '.csv')
        
        if not csv_path.exists():
            if 'slow_' in filename_base:
                fixed_name = filename_base.replace('slow_', 'sow_')
                csv_path_corrected = csv_root / technique_folder / (fixed_name + '.csv')
                if csv_path_corrected.exists():
                    csv_path = csv_path_corrected
                else:
                    continue
            else:
                continue

        try:
            audio, _ = librosa.load(str(wav_path), sr=args.sr, mono=False)
            
            if audio.ndim > 1:
                duration = audio.shape[1] / args.sr
            else:
                duration = len(audio) / args.sr

            df_frame = pd.read_csv(csv_path, skipinitialspace=True)
            df_frame.columns = [col.strip() for col in df_frame.columns]
            
            f0_frames = resample_f0_to_hop(
                df_frame['Time (second)'].values, 
                df_frame['F0'].values, 
                duration, 
                10
            )
            
            out_split_dir = out_root / split
            out_split_dir.mkdir(parents=True, exist_ok=True)
            
            np.savetxt(out_split_dir / (filename_base + '_p.pv'), f0_frames, fmt="%.6f")
            sf.write(out_split_dir / (filename_base + '_p.wav'), audio.T, args.sr, 'PCM_24')

        except Exception as e:
            print(f"Error processing {filename_base}: {e}")
            continue

if __name__ == '__main__':
    args = parse_args()
    process_dataset(args)