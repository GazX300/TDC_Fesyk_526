import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.signal import convolve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

SAMPLE_RATE = 48000
SAMPLE_WIDTH = 2

NAME_ORIGINAL_WAV = f"./Sounds/Sound_{SAMPLE_RATE}[Hz]_{SAMPLE_WIDTH}[byte].wav"

def wavelet_denoiser(signal, level=5, mode='hard', wavelet='db4'):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(signal.size))
    denoised_coeffs = [coeffs[0]] + [
        pywt.threshold(c, threshold, mode=mode) for c in coeffs[1:]
    ]
    denoised_signal = pywt.waverec(denoised_coeffs, wavelet)
    return denoised_signal[:len(signal)]


def gaussian_kernel(size, sigma):
    x = np.linspace(-(size // 2), size // 2, size)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


if __name__ == "__main__":
    data_original, fs_original = sf.read(NAME_ORIGINAL_WAV)
    if len(data_original.shape) > 1:
        data_original = data_original[:, 0]

    P_signal = np.mean(data_original ** 2)

    snr_range = np.arange(-10, 21, 1)

    mse_wt_all, mse_gf_all = [], []
    mae_wt_all, mae_gf_all = [], []
    rmse_wt_all, rmse_gf_all = [], []
    r2_wt_all, r2_gf_all = [], []
    d_wt_all, d_gf_all = [], []

    plot_snr = []
    plot_mse_wt, plot_mse_gf = [], []

    kernel = gaussian_kernel(size=11, sigma=2)

    for snr_db in snr_range:
        P_noise = P_signal / (10 ** (snr_db / 10))

        sigma_noise = np.sqrt(P_noise)

        t_mse_wt, t_mse_gf = [], []
        t_mae_wt, t_mae_gf = [], []
        t_rmse_wt, t_rmse_gf = [], []
        t_r2_wt, t_r2_gf = [], []
        t_d_wt, t_d_gf = [], []

        for _ in range(5):
            noise = np.random.normal(0, sigma_noise, len(data_original))
            data_noisy = data_original + noise

            filtered_wt = wavelet_denoiser(data_noisy, level=5, mode='soft', wavelet='db4')
            filtered_gf = convolve(data_noisy, kernel, mode='same')

            mse_wt = mean_squared_error(data_original, filtered_wt)
            t_mse_wt.append(mse_wt)
            t_mae_wt.append(mean_absolute_error(data_original, filtered_wt))
            t_rmse_wt.append(np.sqrt(mse_wt))
            t_r2_wt.append(r2_score(data_original, filtered_wt))
            t_d_wt.append(np.var(data_original - filtered_wt))

            mse_gf = mean_squared_error(data_original, filtered_gf)
            t_mse_gf.append(mse_gf)
            t_mae_gf.append(mean_absolute_error(data_original, filtered_gf))
            t_rmse_gf.append(np.sqrt(mse_gf))
            t_r2_gf.append(r2_score(data_original, filtered_gf))
            t_d_gf.append(np.var(data_original - filtered_gf))

            plot_snr.append(snr_db)
            plot_mse_wt.append(mse_wt)
            plot_mse_gf.append(mse_gf)

        mse_wt_all.append(np.mean(t_mse_wt))
        mse_gf_all.append(np.mean(t_mse_gf))
        mae_wt_all.append(np.mean(t_mae_wt))
        mae_gf_all.append(np.mean(t_mae_gf))
        rmse_wt_all.append(np.mean(t_rmse_wt))
        rmse_gf_all.append(np.mean(t_rmse_gf))
        r2_wt_all.append(np.mean(t_r2_wt))
        r2_gf_all.append(np.mean(t_r2_gf))
        d_wt_all.append(np.mean(t_d_wt))
        d_gf_all.append(np.mean(t_d_gf))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].scatter(plot_snr, plot_mse_wt, color='blue', alpha=0.15, label="Окремі значення MSE (WT)")
    axes[0].scatter(plot_snr, plot_mse_gf, color='orange', alpha=0.15, label="Окремі значення MSE (GF)")

    axes[0].plot(snr_range, mse_wt_all, color='blue', linewidth=2, label="Середнє MSE WT")
    axes[0].plot(snr_range, mse_gf_all, color='orange', linewidth=2, label="Середнє MSE GF")

    axes[0].set_title("Лінійний масштаб")
    axes[0].set_xlabel("SNR (дБ)")
    axes[0].set_ylabel("MSE")
    axes[0].set_xticks(np.arange(-10, 21, 2))
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].legend()

    axes[1].scatter(plot_snr, plot_mse_wt, color='blue', alpha=0.15, label="Окремі значення MSE (WT)")
    axes[1].scatter(plot_snr, plot_mse_gf, color='orange', alpha=0.15, label="Окремі значення MSE (GF)")

    axes[1].plot(snr_range, mse_wt_all, color='blue', linewidth=2, label="Середня MSE WT")
    axes[1].plot(snr_range, mse_gf_all, color='orange', linewidth=2, label="Середнє MSE GF")

    axes[1].set_yscale('log')
    axes[1].set_title("Логарифмічний масштаб")
    axes[1].set_xlabel("SNR (дБ)")
    axes[1].set_ylabel("MSE (Log Scale)")
    axes[1].set_xticks(np.arange(-10, 21, 2))
    axes[1].grid(True, which="both", linestyle='--', alpha=0.7)
    axes[1].legend()

    plt.tight_layout()

    plt.savefig("./Sounds/snr_vs_mse_comparison.png", dpi=600, bbox_inches='tight')
    plt.show()

    print("\nПорівняльна таблиця завадостійкості (Вибрані опорні точки):")
    print(f"{'SNR (дБ)':<10} | {'MSE Wavelet':<15} | {'MSE Gaussian':<15} | {'R2 Wavelet':<12} | {'R2 Gaussian':<12}")
    print("-" * 75)
    for idx, snr_val in enumerate(snr_range):
        if snr_val in [-10, -5, 0, 5, 10, 15, 20]:  
            print(
                f"{snr_val:<10} | {mse_wt_all[idx]:<15.3e} | {mse_gf_all[idx]:<15.3e} | {r2_wt_all[idx]:<12.4f} | {r2_gf_all[idx]:<12.4f}")