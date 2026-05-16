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

NAME_FILTERED_WAV = "./Sounds/Filtered_4000[Hz]_2[byte].wav"
NAME_FILTERED_RAW = "./Sounds/Filtered_4000[Hz]_2[byte].raw"


def sound_recoder(rec, mic):
    with mic as source:
        print("Говоріть...")
        audio = rec.listen(source)

    wav_data = audio.get_wav_data(
        convert_rate=SAMPLE_RATE,
        convert_width=SAMPLE_WIDTH
    )
    raw_data = audio.get_raw_data(
        convert_rate=SAMPLE_RATE,
        convert_width=SAMPLE_WIDTH
    )

    with open(NAME_ORIGINAL_WAV, "wb") as f:
        f.write(wav_data)

    with open(NAME_ORIGINAL_RAW, "wb") as f:
        f.write(raw_data)


def wavelet_denoiser(signal, level, mode, wavelet):
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


def sound_filter():
    data, fs_original = sf.read(NAME_ORIGINAL_WAV)
    time = np.arange(len(data)) / fs_original

    data_2d = data.reshape(1, -1)

    invariance = denoise_invariant(data_2d, denoise_function=invarince_denoiser).flatten()
    total_variation = denoise_tv_chambolle(data_2d, weight=0.1, channel_axis=None).flatten()
    bilateral = denoise_bilateral(data_2d, sigma_color=0.05, sigma_spatial=15, channel_axis=None).flatten()
    wavelet = wavelet_denoiser(data, level=5, mode='soft', wavelet='db4')

    sf.write("./Sounds/Filtered_Invariance.wav", invariance, SAMPLE_RATE)
    sf.write("./Sounds/Filtered_Total_Variation.wav", total_variation, SAMPLE_RATE)
    sf.write("./Sounds/Filtered_Bilateral.wav", bilateral, SAMPLE_RATE)
    sf.write("./Sounds/Filtered_Wavelet.wav", wavelet, SAMPLE_RATE)

    results = [
        (invariance, "J-Invariance", "Filtered_Invariance.png"),
        (total_variation, "Total Variation", "Filtered_TV.png"),
        (bilateral, "Bilateral Filter", "Filtered_Bilateral.png"),
        (wavelet, "Wavelet Denoising", "Filtered_Wavelet.png")
    ]

    for filtered_data, title, filename in results:
        plt.figure(figsize=(10, 6))
        plt.plot(time, data, 'b', alpha=0.5, label='Original Clean Signal')
        plt.plot(time, filtered_data, 'g', linewidth=2, label=title)

        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)

        plt.savefig(f"./Sounds/{filename}", dpi=300)
        plt.close()


def to_scientific_pretty(x, precision=2):
    superscripts = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
    mantissa, exponent = f"{x:.{precision}e}".split('e')
    mantissa = mantissa.rstrip('0').rstrip('.')
    return f"{mantissa}·10{str(int(exponent)).translate(superscripts)}"


if __name__ == "__main__":
    # sound_filter()
    # recognizer = srec.Recognizer()
    # microphone = srec.Microphone(device_index=1, sample_rate=SAMPLE_RATE)
    # sound_recoder(recognizer, microphone)

    results = []
    row = []
    headers = ['MSE', 'MAE', 'RMSE', 'R2', 'D']

    data_original, fs_original = sf.read(NAME_ORIGINAL_WAV)

    wav_files = glob.glob("./Sounds/*.wav")

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
            to_scientific_pretty(mse),
            to_scientific_pretty(mae),
            to_scientific_pretty(rmse),
            str(round(r2, 2)),
            to_scientific_pretty(D)
        ])

    n_rows = len(row)
    n_cols = len(headers)

    fig, ax = plt.subplots(figsize=(n_cols * 2.8, n_rows * 0.4))
    ax.axis('off')

    table = ax.table(
        cellText=results,
        rowLabels=row,
        colLabels=headers,
        loc='center',
        bbox=[0.08, 0, 1, 1]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)

    plt.savefig("./Sounds/metrics_table.png", dpi=600, bbox_inches='tight')
    plt.show()