import speech_recognition as srec
import soundfile as sf
from math import gcd
from scipy.signal import resample_poly, butter, sosfiltfilt, convolve, resample
import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import denoise_wavelet, denoise_invariant, denoise_tv_chambolle, denoise_bilateral, cycle_spin
import pywt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import glob

SAMPLE_RATE = 48000
SAMPLE_WIDTH = 2
DTYPE = np.int16

NAME_ORIGINAL_WAV = f"./Sounds/Sound_{SAMPLE_RATE}[Hz]_{SAMPLE_WIDTH}[byte].wav"
NAME_ORIGINAL_RAW = f"./Sounds/Sound_{SAMPLE_RATE}[Hz]_{SAMPLE_WIDTH}[byte].raw"

NAME_RESAMPLED_WAV = "./Sounds/Sound_4000[Hz]_2[byte].wav"
NAME_RESAMPLED_RAW = "./Sounds/Sound_4000[Hz]_2[byte].raw"

def wavelet_denoiser(signal, level=5, mode='hard', wavelet='db4'):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(signal.size))
    denoised_coeffs = [coeffs[0]] + [
        pywt.threshold(c, threshold, mode=mode) for c in coeffs[1:]
    ]
    denoised_signal = pywt.waverec(denoised_coeffs, wavelet)
    return denoised_signal[:len(signal)]


def invarince_denoiser(image, **kwargs):
    return denoise_wavelet(image, sigma=0.05, wavelet='db4', mode='soft')


def gaussian_kernel(size, sigma):
    x = np.linspace(-(size // 2), size // 2, size)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def run_all_filters():
    data, fs_original = sf.read(NAME_ORIGINAL_WAV)
    if len(data.shape) > 1:
        data = data[:, 0]

    data_2d = data.reshape(1, -1)
    invariance = denoise_invariant(data_2d, denoise_function=invarince_denoiser).flatten()
    total_variation = denoise_tv_chambolle(data_2d, weight=0.1, channel_axis=None).flatten()
    bilateral = denoise_bilateral(data_2d, sigma_color=0.05, sigma_spatial=15, channel_axis=None).flatten()
    wavelet = wavelet_denoiser(data, level=5, mode='soft', wavelet='db4')

    sf.write("./Sounds/Filtered_Invariance.wav", invariance, SAMPLE_RATE)
    sf.write("./Sounds/Filtered_Total_Variation.wav", total_variation, SAMPLE_RATE)
    sf.write("./Sounds/Filtered_Bilateral.wav", bilateral, SAMPLE_RATE)
    sf.write("./Sounds/Filtered_Wavelet.wav", wavelet, SAMPLE_RATE)

    cutoff = 4000
    sos = butter(6, cutoff, btype='low', fs=SAMPLE_RATE, output='sos')
    filtered_lpf = sosfiltfilt(sos, data)
    sf.write("./Sounds/Filtered_4000[Hz]_2[byte].wav", filtered_lpf, SAMPLE_RATE)

    max_shifts = [0, 1, 3, 5]
    for n, s in enumerate(max_shifts):
        sig_filtered = cycle_spin(data, func=wavelet_denoiser, max_shifts=s, shift_steps=5)
        sf.write(f"./Sounds/Filtered_Shifted_Wavelet_{n}.wav", sig_filtered, SAMPLE_RATE)

    kernel = gaussian_kernel(size=11, sigma=2)
    filtered_gaussian = convolve(data, kernel, mode='same')
    sf.write("./Sounds/Filtered_Gaussian_Filter.wav", filtered_gaussian, SAMPLE_RATE)



def to_latex_scientific(x, precision=2):
    if float(f"{x:.{precision}e}".split('e')[1]) == 0:
        return f"{round(x, precision)}"
    mantissa, exponent = f"{x:.{precision}e}".split('e')
    mantissa = mantissa.rstrip('0').rstrip('.')
    return f"${mantissa} \\cdot 10^{{{int(exponent)}}}$"


if __name__ == "__main__":

    #run_all_filters()

    results = []
    row = []
    headers = ['MSE', 'MAE', 'RMSE', 'R2', 'D']

    data_original, fs_original = sf.read(NAME_ORIGINAL_WAV)
    if len(data_original.shape) > 1:
        data_original = data_original[:, 0]

    wav_files = glob.glob("./Sounds/*.wav")

    wav_files.sort()

    for sounds in wav_files:
        sounds = sounds.replace("\\", "/")

        if sounds == NAME_ORIGINAL_WAV.replace("\\", "/"):
            continue

        elif sounds == NAME_RESAMPLED_WAV.replace("\\", "/"):
            row.append('Ресемпл 4 кГц')
            data, fs = sf.read(sounds)
            data = resample(data, len(data_original))

        else:
            type_filter = sounds.replace('./Sounds/Filtered_', '')
            type_filter = type_filter.replace('.wav', '')
            type_filter = type_filter.replace('_', ' ')

            if type_filter == '4000[Hz] 2[byte]':
                type_filter = 'Лінійний фільтр 4 кГц'
            elif 'Shifted Wavelet' in type_filter:
                type_filter = type_filter.title()

            row.append(type_filter)
            data, fs = sf.read(sounds)

        if len(data) > len(data_original):
            data = data[:len(data_original)]
        elif len(data) < len(data_original):
            data = np.pad(data, (0, len(data_original) - len(data)), 'constant')

        mse = mean_squared_error(data_original, data)
        mae = mean_absolute_error(data_original, data)
        rmse = np.sqrt(mse)
        r2 = r2_score(data_original, data)
        D = np.var(data_original - data)

        results.append([
            to_latex_scientific(mse),
            to_latex_scientific(mae),
            to_latex_scientific(rmse),
            str(round(r2, 2)),
            to_latex_scientific(D)
        ])

    n_rows = len(row)
    n_cols = len(headers)

    fig, ax = plt.subplots(figsize=(n_cols * 2.5, n_rows * 0.38))
    ax.axis('off')

    table = ax.table(
        cellText=results,
        rowLabels=row,
        colLabels=headers,
        loc='center',
        cellLoc='center',
        bbox=[0.08, 0, 1, 1]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)

    for position, cell in table.get_celld().items():
        cell.set_height(0.08)

    plt.savefig("./Sounds/metrics_table.png", dpi=600, bbox_inches='tight') # [cite: 86]
    plt.show()