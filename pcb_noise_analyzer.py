"""
PCB Noise Analyzer — DFT/FFT-Based Signal Analysis Tool
Author: Mellamputi Sai Sushma
Description: Analyzes low-frequency noise in power PCBs using Discrete Fourier Transform (DFT)
             and Fast Fourier Transform (FFT). Identifies dominant noise frequencies,
             harmonic distortion, and signal integrity issues.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import welch, find_peaks
import csv
import os


# ─────────────────────────────────────────────
#  Signal Generation (simulates PCB acquisition)
# ─────────────────────────────────────────────

def generate_pcb_signal(duration=1.0, fs=10000, noise_level=0.3,
                         fundamental=50, harmonics=[150, 250]):
    """
    Simulate a PCB power rail signal with fundamental frequency,
    harmonic components, and random noise.

    Args:
        duration    : Signal duration in seconds
        fs          : Sampling frequency (Hz)
        noise_level : Amplitude of Gaussian noise
        fundamental : Fundamental frequency (Hz), typically 50/60 Hz mains
        harmonics   : List of harmonic frequencies to add

    Returns:
        t  : Time array
        signal : Composite signal array
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    signal = np.sin(2 * np.pi * fundamental * t)  # Fundamental

    for i, h in enumerate(harmonics):
        amplitude = 0.5 / (i + 2)
        signal += amplitude * np.sin(2 * np.pi * h * t + np.random.uniform(0, np.pi))

    signal += noise_level * np.rng.standard_normal(len(t))
    return t, signal


def load_signal_from_csv(filepath):
    """
    Load real oscilloscope/signal analyzer data from CSV.
    CSV format: time(s), voltage(V)

    Args:
        filepath : Path to CSV file

    Returns:
        t      : Time array
        signal : Voltage array
        fs     : Estimated sampling frequency
    """
    t, signal = [], []
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            t.append(float(row[0]))
            signal.append(float(row[1]))

    t = np.array(t)
    signal = np.array(signal)
    fs = 1.0 / np.mean(np.diff(t))
    return t, signal, fs


# ─────────────────────────────────────────────
#  Core Analysis Functions
# ─────────────────────────────────────────────

def compute_fft(signal, fs):
    """
    Compute FFT of the signal and return frequency + magnitude spectrum.

    Args:
        signal : Input signal array
        fs     : Sampling frequency (Hz)

    Returns:
        freqs  : Frequency array (positive side only)
        magnitude : Magnitude spectrum (dB)
    """
    N = len(signal)
    fft_vals = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    magnitude = 20 * np.log10(np.abs(fft_vals) / N + 1e-12)  # dBV
    return freqs, magnitude


def compute_psd(signal, fs, nperseg=256):
    """
    Compute Power Spectral Density using Welch's method.
    Useful for characterizing low-frequency noise floors.

    Args:
        signal  : Input signal
        fs      : Sampling frequency
        nperseg : Segment length for Welch averaging

    Returns:
        freqs : Frequency array
        psd   : Power spectral density (dB/Hz)
    """
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    psd_db = 10 * np.log10(psd + 1e-30)
    return freqs, psd_db


def detect_dominant_frequencies(freqs, magnitude, n_peaks=5, min_prominence=10):
    """
    Detect dominant noise frequencies from the FFT magnitude spectrum.

    Args:
        freqs          : Frequency array
        magnitude      : Magnitude spectrum (dB)
        n_peaks        : Number of top peaks to detect
        min_prominence : Minimum peak prominence in dB

    Returns:
        peak_freqs : List of dominant frequencies
        peak_mags  : Corresponding magnitudes
    """
    peaks, props = find_peaks(magnitude, prominence=min_prominence)
    if len(peaks) == 0:
        return [], []

    # Sort by prominence
    sorted_idx = np.argsort(props['prominences'])[::-1]
    top_peaks = peaks[sorted_idx[:n_peaks]]

    peak_freqs = freqs[top_peaks]
    peak_mags = magnitude[top_peaks]
    return peak_freqs.tolist(), peak_mags.tolist()


def compute_thd(signal, fs, fundamental=50, n_harmonics=5):
    """
    Compute Total Harmonic Distortion (THD) of the signal.

    Args:
        signal      : Input signal
        fs          : Sampling frequency
        fundamental : Fundamental frequency (Hz)
        n_harmonics : Number of harmonics to consider

    Returns:
        thd_percent : THD as a percentage
    """
    fft_vals = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / fs)

    def get_magnitude_at(freq, tolerance=2):
        idx = np.argmin(np.abs(freqs - freq))
        return np.abs(fft_vals[idx])

    v1 = get_magnitude_at(fundamental)
    harmonics_rms = np.sqrt(sum(
        get_magnitude_at(fundamental * n) ** 2
        for n in range(2, n_harmonics + 1)
    ))

    if v1 == 0:
        return 0.0
    thd_percent = (harmonics_rms / v1) * 100
    return round(thd_percent, 3)


def snr_estimate(signal, fs, signal_band=(40, 60), noise_band=(1000, 2000)):
    """
    Estimate Signal-to-Noise Ratio (SNR) by comparing power in
    signal band vs noise band.

    Args:
        signal      : Input signal
        fs          : Sampling frequency
        signal_band : Frequency range of signal (Hz)
        noise_band  : Frequency range of noise floor (Hz)

    Returns:
        snr_db : Estimated SNR in dB
    """
    freqs, psd = welch(signal, fs=fs, nperseg=512)

    def band_power(f_low, f_high):
        mask = (freqs >= f_low) & (freqs <= f_high)
        return np.trapz(psd[mask], freqs[mask])

    sig_power = band_power(*signal_band)
    noise_power = band_power(*noise_band)

    if noise_power == 0:
        return float('inf')
    return round(10 * np.log10(sig_power / noise_power), 2)


# ─────────────────────────────────────────────
#  Visualization
# ─────────────────────────────────────────────

def plot_full_analysis(t, signal, freqs_fft, mag_fft,
                        freqs_psd, psd, peak_freqs, peak_mags,
                        title="PCB Noise Analysis Report"):
    """
    Generate a comprehensive 4-panel analysis plot.
    """
    fig = plt.figure(figsize=(14, 10), facecolor='#0d1117')
    fig.suptitle(title, color='white', fontsize=14, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    panel_cfg = dict(facecolor='#161b22')
    text_cfg  = dict(color='#c9d1d9')
    grid_cfg  = dict(color='#30363d', linestyle='--', linewidth=0.5)

    # ── Panel 1: Time Domain ──
    ax1 = fig.add_subplot(gs[0, 0], **panel_cfg)
    ax1.plot(t[:2000] * 1000, signal[:2000], color='#58a6ff', linewidth=0.8)
    ax1.set_title('Time Domain Signal', **text_cfg)
    ax1.set_xlabel('Time (ms)', **text_cfg)
    ax1.set_ylabel('Voltage (V)', **text_cfg)
    ax1.tick_params(colors='#8b949e')
    ax1.grid(**grid_cfg)
    for spine in ax1.spines.values():
        spine.set_edgecolor('#30363d')

    # ── Panel 2: FFT Spectrum ──
    ax2 = fig.add_subplot(gs[0, 1], **panel_cfg)
    ax2.plot(freqs_fft, mag_fft, color='#3fb950', linewidth=0.8)
    if peak_freqs:
        ax2.scatter(peak_freqs, peak_mags, color='#f85149', zorder=5,
                    s=40, label='Dominant freqs')
        for pf, pm in zip(peak_freqs, peak_mags):
            ax2.annotate(f'{pf:.0f}Hz', (pf, pm),
                         textcoords="offset points", xytext=(4, 4),
                         fontsize=7, color='#ffa657')
    ax2.set_title('FFT Magnitude Spectrum', **text_cfg)
    ax2.set_xlabel('Frequency (Hz)', **text_cfg)
    ax2.set_ylabel('Magnitude (dBV)', **text_cfg)
    ax2.set_xlim(0, min(2000, freqs_fft[-1]))
    ax2.tick_params(colors='#8b949e')
    ax2.grid(**grid_cfg)
    ax2.legend(facecolor='#21262d', labelcolor='#c9d1d9', fontsize=8)
    for spine in ax2.spines.values():
        spine.set_edgecolor('#30363d')

    # ── Panel 3: Power Spectral Density ──
    ax3 = fig.add_subplot(gs[1, 0], **panel_cfg)
    ax3.semilogy(freqs_psd, 10 ** (psd / 10), color='#d2a8ff', linewidth=0.8)
    ax3.set_title('Power Spectral Density (Welch)', **text_cfg)
    ax3.set_xlabel('Frequency (Hz)', **text_cfg)
    ax3.set_ylabel('PSD (V²/Hz)', **text_cfg)
    ax3.tick_params(colors='#8b949e')
    ax3.grid(**grid_cfg)
    for spine in ax3.spines.values():
        spine.set_edgecolor('#30363d')

    # ── Panel 4: Noise Statistics ──
    ax4 = fig.add_subplot(gs[1, 1], **panel_cfg)
    ax4.hist(signal, bins=60, color='#ffa657', edgecolor='#30363d',
             linewidth=0.4, alpha=0.85)
    ax4.set_title('Noise Amplitude Distribution', **text_cfg)
    ax4.set_xlabel('Amplitude (V)', **text_cfg)
    ax4.set_ylabel('Count', **text_cfg)
    ax4.tick_params(colors='#8b949e')
    ax4.grid(**grid_cfg)
    for spine in ax4.spines.values():
        spine.set_edgecolor('#30363d')

    plt.savefig('pcb_analysis_report.png', dpi=150, bbox_inches='tight',
                facecolor='#0d1117')
    plt.show()
    print("📊 Report saved → pcb_analysis_report.png")


# ─────────────────────────────────────────────
#  Main Entry Point
# ─────────────────────────────────────────────

def analyze(filepath=None, fs=10000, fundamental=50):
    """
    Run full PCB noise analysis pipeline.

    Args:
        filepath   : Path to CSV file (None = use simulated signal)
        fs         : Sampling frequency (Hz)
        fundamental: Fundamental frequency for THD calculation
    """
    print("=" * 55)
    print("   PCB Noise Analyzer — DFT/FFT Analysis Tool")
    print("=" * 55)

    # Load or generate signal
    if filepath and os.path.exists(filepath):
        print(f"📂 Loading signal from: {filepath}")
        t, signal, fs = load_signal_from_csv(filepath)
    else:
        print("⚡ Using simulated PCB power rail signal (50 Hz + harmonics + noise)")
        rng = np.random.default_rng(42)
        t = np.linspace(0, 1.0, int(fs * 1.0), endpoint=False)
        signal = (np.sin(2 * np.pi * 50 * t) +
                  0.25 * np.sin(2 * np.pi * 150 * t + 0.5) +
                  0.15 * np.sin(2 * np.pi * 250 * t + 1.2) +
                  0.3 * rng.standard_normal(len(t)))

    # Analysis
    freqs_fft, mag_fft   = compute_fft(signal, fs)
    freqs_psd, psd        = compute_psd(signal, fs)
    peak_freqs, peak_mags = detect_dominant_frequencies(freqs_fft, mag_fft)
    thd                   = compute_thd(signal, fs, fundamental=fundamental)
    snr                   = snr_estimate(signal, fs)

    # Report
    print(f"\n📋 Analysis Results")
    print(f"   Sampling Rate      : {fs:.0f} Hz")
    print(f"   Signal Duration    : {t[-1]:.3f} s")
    print(f"   Signal RMS         : {np.sqrt(np.mean(signal**2)):.4f} V")
    print(f"   Peak Amplitude     : {np.max(np.abs(signal)):.4f} V")
    print(f"   THD                : {thd:.2f} %")
    print(f"   Estimated SNR      : {snr:.2f} dB")
    print(f"\n   Dominant Noise Frequencies:")
    for f, m in zip(peak_freqs, peak_mags):
        print(f"     → {f:.1f} Hz  ({m:.1f} dBV)")

    # Plot
    plot_full_analysis(t, signal, freqs_fft, mag_fft,
                        freqs_psd, psd, peak_freqs, peak_mags)


if __name__ == "__main__":
    analyze()
