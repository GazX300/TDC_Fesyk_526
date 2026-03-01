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

discrete_signals = []
discrete_spectrums = []
restored_signals = []
variances = []
snr_values = []

var_original = np.var(filtered_signal)

for Dt in [2, 4, 8, 16]:

    discrete_signal = np.zeros(n)

    for i in range(0, round(n / Dt)):
        discrete_signal[i * Dt] = filtered_signal[i * Dt]

    discrete_signals += [list(discrete_signal)]

    #спектр дискретного сигналу
    spectrum = fft.fft(discrete_signal)
    spectrum_shifted = np.abs(fft.fftshift(spectrum)) / n
    discrete_spectrums += [list(spectrum_shifted)]

    w = 22 / (Fs / 2)   # F_filter = 22 Гц для варіанту 7
    sos = signal.butter(3, w, 'low', output='sos')

    restored = signal.sosfiltfilt(sos, discrete_signal)
    restored_signals += [list(restored)]
    #похибка
    error = restored - filtered_signal
    var_error = np.var(error)

    variances += [var_error]
    snr_values += [var_original / var_error]

def plot_signal(x, y, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(21/2.54, 14/2.54))
    ax.plot(x, y, linewidth=1)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=14)
    ax.grid(True)
    fig.savefig('./figures/' + title + '.png', dpi=600)
    plt.close(fig)

fig, ax = plt.subplots(2, 2, figsize=(21/2.54, 14/2.54))

s = 0
for i in range(0, 2):
    for j in range(0, 2):
        ax[i][j].plot(time, discrete_signals[s], linewidth=1)
        ax[i][j].set_title(f'Dt = {[2,4,8,16][s]}', fontsize=14)
        ax[i][j].grid(True)
        s += 1

fig.supxlabel("Час (секунди)", fontsize=14)
fig.supylabel("Амплітуда", fontsize=14)
fig.suptitle("Дискретизовані сигнали", fontsize=14)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig('./figures/Дискретизований сигнал.png', dpi=600)
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

fig, ax = plt.subplots(2, 2, figsize=(21/2.54, 14/2.54))

s = 0
for i in range(0, 2):
    for j in range(0, 2):
        ax[i][j].plot(freqs_shifted, discrete_spectrums[s], linewidth=1)
        ax[i][j].set_title(f'Dt = {[2,4,8,16][s]}', fontsize=14)
        ax[i][j].grid(True)
        s += 1

fig.supxlabel("Частота (Гц)", fontsize=14)
fig.supylabel("Амплітуда спектру", fontsize=14)
fig.suptitle("Спектри дискретизованих сигналів", fontsize=14)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig('./figures/Спектр дискретизованого сигналу.png', dpi=600)
plt.close(fig)

fig, ax = plt.subplots(2, 2, figsize=(21/2.54, 14/2.54))

s = 0
for i in range(0, 2):
    for j in range(0, 2):
        ax[i][j].plot(time, restored_signals[s], linewidth=1)
        ax[i][j].set_title(f'Dt = {[2,4,8,16][s]}', fontsize=14)
        ax[i][j].grid(True)
        s += 1

fig.supxlabel("Час (секунди)", fontsize=14)
fig.supylabel("Амплітуда", fontsize=14)
fig.suptitle("Відновлені сигнали", fontsize=14)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig('./figures/Відновлений сигнал.png', dpi=600)
plt.close(fig)

Dt_values = [2, 4, 8, 16]

plot_signal(Dt_values, variances,
            "Залежність дисперсії від кроку дискретизації",
            "Крок дискретизації",
            "Дисперсія")

plot_signal(Dt_values, snr_values,
            "Залежність SNR від кроку дискретизації",
            "Крок дискретизації",
            "SNR")
