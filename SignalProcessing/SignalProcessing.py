import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, fft

#варіант 7
n = 500
Fs = 1000
F_max = 15

random_signal = np.random.normal(0, 10, n)
time = np.arange(n) / Fs
w = F_max / (Fs / 2)
sos = signal.butter(3, w, 'low', output='sos')

filtered_signal = signal.sosfiltfilt(sos, random_signal)

def plot_signal(x, y, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(21/2.54, 14/2.54))
    ax.plot(x, y, linewidth=1)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=14)
    ax.grid(True)
    fig.savefig('./figures/' + title + '.png', dpi=600)
    plt.close(fig)


plot_signal(time, filtered_signal,
            "Сигнал з максимальною частотою 15Гц",
            "Час (секунди)",
            "Амплітуда")

spectrum = fft.fft(filtered_signal)
spectrum_shifted = np.abs(fft.fftshift(spectrum))

freqs = fft.fftfreq(n, 1/Fs)
freqs_shifted = fft.fftshift(freqs)

plot_signal(freqs_shifted, spectrum_shifted,
            "Спектр сигналу",
            "Частота",
            "Амплітуда спектру")