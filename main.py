import speech_recognition as srec
import soundfile as sf
from math import gcd
from scipy.signal import resample_poly, butter, sosfiltfilt
import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import denoise_wavelet, denoise_invariant, denoise_tv_chambolle, denoise_bilateral
import pywt

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
        plt.show()


if __name__ == "__main__":
    sound_filter()

    recognizer = srec.Recognizer()
    microphone = srec.Microphone(device_index=1, sample_rate=SAMPLE_RATE)

    sound_recoder(recognizer, microphone)

    #ресемплінг

    data, fs_original = sf.read(NAME_ORIGINAL_WAV)

    fs_target = 4000

    g = gcd(fs_original, fs_target)
    up = fs_target // g
    down = fs_original // g

    data_resampled = resample_poly(data, up, down)

    sf.write(NAME_RESAMPLED_WAV, data_resampled, fs_target)


    with open(NAME_ORIGINAL_RAW, "rb") as f:
        raw_bytes = f.read()

    signal = np.frombuffer(raw_bytes, dtype=DTYPE)

    signal_float = signal.astype(np.float32) / 32768.0

    resampled = resample_poly(signal_float, up, down)

    resampled_int16 = np.int16(resampled * 32767)

    with open(NAME_RESAMPLED_RAW, "wb") as f:
        f.write(resampled_int16.tobytes())

    #Фільтр

    data, fs_original = sf.read(NAME_ORIGINAL_WAV)

    if len(data.shape) > 1:
        data = data[:, 0]

    cutoff = 4000
    order = 6

    sos = butter(order, cutoff, btype='low', fs=SAMPLE_RATE, output='sos')

    filtered = sosfiltfilt(sos, data)

    sf.write(NAME_FILTERED_WAV, filtered, SAMPLE_RATE)


    with open(NAME_ORIGINAL_RAW, "rb") as f:
        raw_bytes = f.read()

    signal = np.frombuffer(raw_bytes, dtype=DTYPE)

    signal_float = signal.astype(np.float32) / 32768.0

    filtered_raw = sosfiltfilt(sos, signal_float)

    filtered_int16 = np.int16(filtered_raw * 32767)

    with open(NAME_FILTERED_RAW, "wb") as f:
        f.write(filtered_int16.tobytes())

    #Графіки

    data, fs = sf.read(NAME_ORIGINAL_WAV)
    time = np.arange(len(data)) / fs

    plt.figure(figsize=(12, 6))
    plt.plot(time, data, label=f"Оригінал (fs={fs} Гц)")

    data, fs = sf.read(NAME_RESAMPLED_WAV)
    time = np.arange(len(data)) / fs
    plt.plot(time, data, label=f"Ресемпл (fs={fs} Гц)")

    data, fs = sf.read(NAME_FILTERED_WAV)
    time = np.arange(len(data)) / fs
    plt.plot(time, data, label=f"Фільтрований (LPF {cutoff} Гц)")

    plt.title("Порівняння сигналів у часовій області")
    plt.xlabel("Час (с)")
    plt.ylabel("Амплітуда")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig("./Sounds/signals_comparison.png", dpi=300)

    plt.show()