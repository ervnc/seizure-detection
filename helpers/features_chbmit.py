# helpers/features_chbmit.py

import os
import numpy as np

# Bandas do EEG (Hz)
EEG_BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}


def _prepare_band_masks(n_samples: int, sfreq: float):
    """
    Pré-calcula os índices de frequências para cada banda, dado
    o tamanho da janela e a frequência de amostragem.
    """
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sfreq)
    band_masks = {}
    for name, (fmin, fmax) in EEG_BANDS.items():
        band_masks[name] = (freqs >= fmin) & (freqs <= fmax)
    return freqs, band_masks


def extract_features_all(raw, windows):
    """
    raw: MNE Raw (já filtrado, reamostrado, só canais EEG)
    windows: array (n_windows, 2) com [start_idx, end_idx]

    Retorna:
        X: (n_windows, n_features)
    """
    data = raw.get_data()  # (n_channels, n_times)
    sfreq = float(raw.info["sfreq"])
    n_channels, _ = data.shape

    windows = np.asarray(windows, dtype=int)
    n_windows = windows.shape[0]
    win_len = windows[0, 1] - windows[0, 0]

    # máscaras de banda pré-computadas para esse tamanho de janela
    _, band_masks = _prepare_band_masks(win_len, sfreq)

    # Número de features por canal:
    #   mean, std, rms + 5 bandpowers = 8
    feats_per_channel = 3 + len(EEG_BANDS)
    n_features = n_channels * feats_per_channel

    X = np.zeros((n_windows, n_features), dtype=np.float32)

    for i, (start, end) in enumerate(windows):
        segment = data[:, start:end]  # (n_channels, win_len)

        # estatísticas simples (vetorizadas)
        mean = segment.mean(axis=1)            # (n_channels,)
        std = segment.std(axis=1)
        rms = np.sqrt((segment ** 2).mean(axis=1))

        # FFT ao longo do tempo (axis=1)
        fft_vals = np.fft.rfft(segment, axis=1)
        psd = (np.abs(fft_vals) ** 2) / win_len   # (n_channels, n_freqs)

        band_feats = []
        for name in EEG_BANDS.keys():
            mask = band_masks[name]
            # soma de potência na banda
            bp = psd[:, mask].sum(axis=1)       # (n_channels,)
            band_feats.append(bp)

        # concatena features por canal
        # shape final por janela: (n_channels * feats_per_channel,)
        feats = np.concatenate(
            [mean, std, rms] + band_feats,
            axis=0
        ).astype(np.float32)

        X[i, :] = feats

    return X
