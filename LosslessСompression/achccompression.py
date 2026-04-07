import math
import collections
import matplotlib.pyplot as plt

def float_bin(point, size_cod):
    binary_code = ""
    for _ in range(size_cod):
        point *= 2
        if point > 1:
            binary_code += "1"
            point -= int(point)
        elif point == 1:
            binary_code += "1"
            break
        else:
            binary_code += "0"
    return binary_code


# А ENCODE
def encode_ac(uniq_chars, probabilitys, alphabet_size, sequence):
    alphabet = list(uniq_chars)
    probability = [probabilitys[s] for s in alphabet]

    unity = []
    probability_range = 0.0

    for i in range(alphabet_size):
        l = probability_range
        probability_range += probability[i]
        u = probability_range
        unity.append([alphabet[i], l, u])

    for i in range(len(sequence) - 1):
        for j in range(len(unity)):
            if sequence[i] == unity[j][0]:
                low = unity[j][1]
                high = unity[j][2]
                diff = high - low

                for k in range(len(unity)):
                    unity[k][1] = low
                    unity[k][2] = probability[k] * diff + low
                    low = unity[k][2]
                break

    low, high = 0, 0
    for i in range(len(unity)):
        if unity[i][0] == sequence[-1]:
            low = unity[i][1]
            high = unity[i][2]

    point = (low + high) / 2
    size_cod = math.ceil(math.log2(1 / (high - low)) + 1)
    bin_code = float_bin(point, size_cod)

    return [point, alphabet_size, alphabet, probability], bin_code


# A DECODE
def decode_ac(encoded_data, length_seq):
    point, alphabet_size, alphabet, probability = encoded_data

    unity = []
    prob_range = 0.0
    for i in range(alphabet_size):
        l = prob_range
        prob_range += probability[i]
        u = prob_range
        unity.append([alphabet[i], l, u])

    decoded = ""

    for _ in range(length_seq):
        for j in range(len(unity)):
            if unity[j][1] < point < unity[j][2]:
                decoded += unity[j][0]
                low = unity[j][1]
                high = unity[j][2]
                diff = high - low

                for k in range(len(unity)):
                    unity[k][1] = low
                    unity[k][2] = probability[k] * diff + low
                    low = unity[k][2]
                break

    return decoded


# HUFFMAN
def encode_ch(uniq_chars, probabilitys, sequence):
    alphabet = list(uniq_chars)
    probability = [probabilitys[s] for s in alphabet]

    final = [[alphabet[i], probability[i]] for i in range(len(alphabet))]
    final.sort(key=lambda x: x[1])

    tree = []

    if len(set(probability)) == 1:
        symbol_code = []
        for i in range(len(alphabet)):
            code = "1" * i + "0"
            symbol_code.append([alphabet[i], code])
        encode = "".join([symbol_code[alphabet.index(c)][1] for c in sequence])
        return [encode, symbol_code], encode

    while len(final) > 1:
        left = final.pop(0)
        right = final.pop(0)
        tot = left[1] + right[1]
        tree.append([left[0], right[0]])
        final.append([left[0] + right[0], tot])
        final.sort(key=lambda x: x[1])

    tree.reverse()
    alphabet.sort()

    symbol_code = []

    for char in alphabet:
        code = ""
        for node in tree:
            if char in node[0]:
                code += "0"
                if char == node[0]:
                    break
            else:
                code += "1"
                if char == node[1]:
                    break
        symbol_code.append([char, code])

    encode = ""
    for c in sequence:
        encode += [x[1] for x in symbol_code if x[0] == c][0]

    return [encode, symbol_code], encode


# HUFFMAN DECODE
def decode_ch(encoded):
    encode = list(encoded[0])
    symbol_code = encoded[1]

    sequence = ""
    temp = ""

    for bit in encode:
        temp += bit
        for sym, code in symbol_code:
            if temp == code:
                sequence += sym
                temp = ""
                break

    return sequence


#MAIN
def main():
    with open("sequence.txt", "r") as f:
        lines = f.readlines()

    sequences = []
    for line in lines:
        line = line.strip()
        if line.startswith("Sequence"):
            seq = line.split(":")[1].strip()
            sequences.append(seq)

    results = []

    open("results_AC_CH.txt", "w", encoding="utf-8").close()

    for idx, seq in enumerate(sequences):
        seq = seq.strip()
        seq_to_encode = seq[:10]  # Кодуємо перші 10 символів

        N = len(seq_to_encode)
        counts = collections.Counter(seq_to_encode)
        uniq = sorted(list(set(seq_to_encode)))

        prob = {s: counts[s] / N for s in uniq}
        entropy = -sum(p * math.log2(p) for p in prob.values())

        # AC
        ac_data, ac_code = encode_ac(uniq, prob, len(uniq), seq_to_encode)
        ac_dec = decode_ac(ac_data, N)
        bps_ac = len(ac_code) / N

        # CH
        ch_data, ch_code_table = encode_ch(uniq, prob, seq_to_encode)
        ch_dec = decode_ch(ch_data)
        bps_ch = len(ch_code_table) / N

        results.append([round(entropy, 2), round(bps_ac, 2), round(bps_ch, 2)])

        with open("results_AC_CH.txt", "a", encoding="utf-8") as f:
            f.write(f"---------- Послідовність {idx + 1} ----------\n")
            f.write(f"Послідовність: {seq_to_encode}\n")
            f.write(f"Ентропія: {round(entropy, 2)}\n\n")

            f.write(f"Арифметичне кодування:\n")
            f.write(f"Закодована послідовність: {ac_code}\n")
            f.write(f"Декодована послідовність: {ac_dec}\n")
            f.write(f"bps: {round(bps_ac, 2)}\n\n")

            f.write(f"Кодування Хаффмана:\n")
            for sym_code in ch_data[1]:
                f.write(f"Символ: {sym_code[0]} Код: {sym_code[1]}\n")
            f.write(f"Закодована послідовність: {ch_data[0]}\n")
            f.write(f"Декодована послідовність: {ch_dec}\n")
            f.write(f"bps: {round(bps_ch, 2)}\n")
            f.write("-" * 40 + "\n\n")

    # TABLE
    N_rows = len(results)
    fig, ax = plt.subplots(figsize=(10, N_rows * 0.8))
    ax.axis('off')
    headers = ['Ентропія', 'bps AC', 'bps CH']
    row_labels = [f'Послідовність {i + 1}' for i in range(N_rows)]

    table = ax.table(
        cellText=results,
        colLabels=headers,
        rowLabels=row_labels,
        loc='center',
        cellLoc='center'
    )

    table.set_fontsize(14)
    table.scale(0.8, 2)

    plt.title("Результати стиснення методами AC та CH", fontsize=16, pad=20)
    plt.savefig("Результати стиснення методами AC та CH.png", bbox_inches='tight', dpi=300)

    plt.show()

if __name__ == "__main__":
    main()