import random
import string
import collections
import math
import matplotlib.pyplot as plt

N_sequence = 100

surname = "Fesyk"
group = "526"
student_number = 7

# --- 1 ---
list1 = ['1'] * student_number
list0 = ['0'] * (N_sequence - student_number)
seq1 = list1 + list0
random.shuffle(seq1)
seq1 = ''.join(seq1)

# --- 2 ---
list1 = list(surname)
list0 = ['0'] * (N_sequence - len(list1))
seq2 = ''.join(list1 + list0)

# --- 3 ---
list1 = list(surname)
list0 = ['0'] * (N_sequence - len(list1))
seq3 = list1 + list0
random.shuffle(seq3)
seq3 = ''.join(seq3)

# --- 4 ---
letters = list(surname) + list(group)
n_letters = len(letters)
n_repeats = N_sequence // n_letters
remainder = N_sequence % n_letters
seq4 = letters * n_repeats + letters[:remainder]
seq4 = ''.join(seq4)

# --- 5 ---
elements = list(surname[:2]) + list(group)
seq5 = ''.join(random.choice(elements) for _ in range(N_sequence))

# --- 6 ---
letters = list(surname[:2])
digits = list(group)

n_letters = int(0.7 * N_sequence)
n_digits = int(0.3 * N_sequence)

seq6_list = []
for _ in range(n_letters):
    seq6_list.append(random.choice(letters))
for _ in range(n_digits):
    seq6_list.append(random.choice(digits))

random.shuffle(seq6_list)
seq6 = ''.join(seq6_list)

# --- 7 ---
elements = string.ascii_lowercase + string.digits
seq7 = ''.join(random.choice(elements) for _ in range(N_sequence))

# --- 8 ---
seq8 = '1' * N_sequence

original_sequences = [seq1, seq2, seq3, seq4, seq5, seq6, seq7, seq8]

with open("sequence.txt", "w") as f:
    for i, seq in enumerate(original_sequences, 1):
        f.write(f"Sequence {i}: {seq}\n\n")

results = []

with open("results_sequence.txt", "w") as file:
    for idx, sequence in enumerate(original_sequences, 1):

        counts = collections.Counter(sequence)
        probability = {s: c / N_sequence for s, c in counts.items()}

        mean_probability = sum(probability.values()) / len(probability)

        equal = all(abs(p - mean_probability) < 0.05 * mean_probability for p in probability.values())
        uniformity = "рівна" if equal else "нерівна"

        entropy = -sum(p * math.log2(p) for p in probability.values())

        alphabet_size = len(counts)

        if alphabet_size > 1:
            source_excess = 1 - entropy / math.log2(alphabet_size)
        else:
            source_excess = 1

        probability_str = ', '.join([f"{k}={v:.4f}" for k, v in probability.items()])

        file.write(f"\nПослідовність {idx}\n")
        file.write(f"Розмір алфавіту: {alphabet_size}\n")
        file.write(f"Ймовірності: {probability_str}\n")
        file.write(f"Ентропія: {entropy:.4f}\n")
        file.write(f"Надмірність: {source_excess:.4f}\n")
        file.write(f"Тип: {uniformity}\n")

        results.append([alphabet_size, round(entropy, 2), round(source_excess, 2), uniformity])

# Таблиця
headers = ['Розмір алфавіту', 'Ентропія', 'Надмірність', 'Ймовірність']
rows = [f'Послідовність {i}' for i in range(1, 9)]

fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')

table = ax.table(cellText=results, colLabels=headers, rowLabels=rows, loc='center' , cellLoc='center')
table.set_fontsize(14)
table.scale(0.8, 2)

plt.savefig("характеристики.png")
plt.show()