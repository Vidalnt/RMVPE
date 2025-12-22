import numpy as np
import torch
from tqdm import tqdm
from scipy.ndimage import find_objects, label
from collections import defaultdict
from src import to_local_average_cents, bce, SAMPLE_RATE
from mir_eval.melody import (
    raw_pitch_accuracy,
    to_cent_voicing,
    raw_chroma_accuracy,
    overall_accuracy,
)
from mir_eval.melody import voicing_recall, voicing_false_alarm
import torch.nn.functional as F
import gc
import matplotlib

matplotlib.use("Agg")
import matplotlib.pylab as plt


def plot_f0_compared(f0, cleanf0, title="F0 Comparison"):
    f0 = np.array(f0)
    cleanf0 = np.array(cleanf0)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(f0, label="hat")
    ax.plot(cleanf0, label="gt")
    ax.set_ylim(0, None)
    ax.set_xlim(0, 500)
    ax.set_title(title)
    ax.legend()
    fig.canvas.draw()
    plt.close(fig)
    return fig


def evaluate_pitch_smoothness(pitch_pred, pred_voicing, true_voicing):
    relative_smoothness, continuity_breaks = np.nan, np.nan
    voiced_idx = np.where(pred_voicing)[0]
    if len(voiced_idx) >= 2:
        consecutive = np.diff(voiced_idx) == 1
        ps, pe = (
            pitch_pred[voiced_idx[:-1][consecutive]],
            pitch_pred[voiced_idx[1:][consecutive]],
        )
        mask = (ps > 0) & (pe > 0)
        if np.any(mask):
            rel_changes = np.abs(pe[mask] - ps[mask]) / (ps[mask] + 1e-8)
            m, s = np.mean(rel_changes), np.std(rel_changes)
            relative_smoothness = s / m if m > 1e-9 else 0.0

    labeled, num_segments = label(true_voicing)
    if num_segments > 0:
        gt_segments = find_objects(labeled)
        total = sum(1 for s in gt_segments if s[0].stop - s[0].start > 1)
        breaks = sum(
            1
            for s in gt_segments
            if s[0].stop - s[0].start > 1 and not np.all(pred_voicing[s[0]])
        )
        continuity_breaks = breaks / total if total > 0 else np.nan
    return {
        "relative_smoothness": float(relative_smoothness),
        "continuity_breaks": float(continuity_breaks),
    }


def process_in_chunks(mel, model, device, chunk_size=32000):
    n_frames = mel.shape[-1]
    mel_padded = F.pad(
        mel, (0, 32 * ((n_frames - 1) // 32 + 1) - n_frames), mode="reflect"
    )
    output_chunks = []
    with torch.no_grad():
        for start in range(0, mel_padded.shape[-1], chunk_size):
            out_chunk = model(
                mel_padded[:, start : start + chunk_size].unsqueeze(0)
            ).squeeze(0)
            output_chunks.append(out_chunk)
    return torch.cat(output_chunks, dim=0)[:n_frames]


def evaluate(dataset, model, hop_length, device, pitch_th=0.03):
    metrics = defaultdict(list)
    figures = []

    def calculate_metrics(pitch_p, pitch_l, file_name):
        loss = bce(pitch_p, pitch_l)
        metrics["loss"].append(loss.item())
        c_pred = to_local_average_cents(pitch_p.cpu().numpy(), None, pitch_th)
        c_true = to_local_average_cents(pitch_l.cpu().numpy(), None, pitch_th)

        f_pred = np.where(c_pred > 0, 10 * (2 ** (c_pred / 1200)), 0)
        f_true = np.where(c_true > 0, 10 * (2 ** (c_true / 1200)), 0)
        t = np.array([i * hop_length * 1000 / SAMPLE_RATE for i in range(len(c_true))])

        rv, rc, ev, ec = to_cent_voicing(t, f_true, t, f_pred)
        metrics["RPA"].append(raw_pitch_accuracy(rv, rc, ev, ec))
        metrics["RCA"].append(raw_chroma_accuracy(rv, rc, ev, ec))
        metrics["OA"].append(overall_accuracy(rv, rc, ev, ec))
        metrics["VFA"].append(voicing_false_alarm(rv, ev))
        metrics["VR"].append(voicing_recall(rv, ev))

        sm = evaluate_pitch_smoothness(f_pred, f_pred > 0, f_true > 0)
        metrics["SMOOTH"].append(sm["relative_smoothness"])
        metrics["BREAKS"].append(sm["continuity_breaks"])

        if len(figures) < 6:
            fig = plot_f0_compared(f_pred, f_true, title=file_name)
            figures.append((file_name, fig))

        print(
            f"{file_name} :\t RPA: {metrics['RPA'][-1]:.4f} \t OA: {metrics['OA'][-1]:.4f}"
        )

    for data in dataset:
        try:
            mel, pitch_label = data["mel"].to(device), data["pitch"].to(device)
            try:
                n = mel.shape[-1]
                mel_p = F.pad(mel, (0, 32 * ((n - 1) // 32 + 1) - n), mode="reflect")
                pitch_pred = model(mel_p.unsqueeze(0)).squeeze(0)[
                    : pitch_label.shape[0]
                ]
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    gc.collect()
                    pitch_pred = process_in_chunks(mel, model, device)
                    pitch_pred = pitch_pred[: pitch_label.shape[0]]
                else:
                    raise e

            calculate_metrics(pitch_pred, pitch_label, data["file"])
            del mel, pitch_pred, pitch_label
        except Exception:
            torch.cuda.empty_cache()
            continue
    return metrics, figures
