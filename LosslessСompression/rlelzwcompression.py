import ast
import collections
import math
import matplotlib.pyplot as plt


#КОДУВАННЯ ТА ДЕКОДУВАННЯ RLE
def encode_rle(sequence):
    if not sequence: return "", []
    count = 1
    result = []

    for i, item in enumerate(sequence):
        if i == 0:
            continue
        if item == sequence[i - 1]:
            count += 1
        else:
            result.append((sequence[i - 1], count))
            count = 1
    result.append((sequence[-1], count))

    # Формування рядка
    encoded = []
    for i, item in enumerate(result):
        encoded.append(f"{item[1]}{item[0]}")

    return "".join(encoded), result


def decode_rle(sequence):
    result = []
    for item in sequence:
        # item[0] - символ, item[1] - кількість повторень
        result.append(item[0] * item[1])
    return "".join(result)


# КОДУВАННЯ ТА ДЕКОДУВАННЯ LZW
def encode_zlw(sequence, file_handle):
    dictionary = {}
    for i in range(65536):
        dictionary[chr(i)] = i

    current = ""
    result = []
    size = 0

    file_handle.write("________Словник________\n")

    for c in sequence:
        new_str = current + c
        if new_str in dictionary:
            current = new_str
        else:
            code = dictionary[current]
            result.append(code)
            element_bits = 16 if code < 65536 else math.ceil(math.log2(len(dictionary)))
            size += element_bits

            file_handle.write(f"Code: {code}, Element: {current}, Bits: {element_bits}\n")

            dictionary[new_str] = len(dictionary)
            current = c

    if current:
        code = dictionary[current]
        result.append(code)
        last = 16 if code < 65536 else math.ceil(math.log2(len(dictionary)))
        size += last
        file_handle.write(f"Code: {code}, Element: {current}, Bits: {last}\n")

    return result, size


def decode_zlw(sequence):
    dictionary = {}
    for i in range(65536):
        dictionary[i] = chr(i)

    result = ""
    previous = None
    current = ""

    for code in sequence:
        if code in dictionary:
            current = dictionary[code]
            result += current
            if previous is not None:
                dictionary[len(dictionary)] = previous + current[0]
            previous = current
        else:
            current = previous + previous[0]
            result += current
            dictionary[len(dictionary)] = current
            previous = current

    return result


def main():
    # 1. Зчитування файлу
    try:
        with open("sequence.txt", "r", encoding="utf-8") as file:
            lines = file.readlines()
            original_sequences = [line.split(":", 1)[1].strip() for line in lines if ":" in line]
    except FileNotFoundError:
        print("Помилка: Файл sequence.txt не знайдено!")
        return

    results_table = []
    N_sequence = 100

    # Відкриваємо файл для запису результатів
    with open("results_rle_lzw.txt", "w", encoding="utf-8") as out_file:
        for seq_idx, sequence in enumerate(original_sequences):
            out_file.write("/" * 80 + "\n")
            out_file.write(f"Оригінальна послідовність: {sequence}\n")

            # Розмір оригінальної послідовності
            orig_size_bits = len(sequence) * 16
            out_file.write(f"Розмір оригінальної послідовості {orig_size_bits} bits\n")

            # Обчислення ентропії
            counts = collections.Counter(sequence)
            probability = {symbol: count / len(sequence) for symbol, count in counts.items()}
            entropy = -sum(p * math.log2(p) for p in probability.values() if p > 0)
            out_file.write(f" Ентропія: {round(entropy, 4)}\n\n")

            #RLE
            out_file.write("________Кодування_RLE________\n")
            encoded_str_rle, encoded_rle_tuples = encode_rle(sequence)
            rle_size_bits = len(encoded_str_rle) * 16

            out_file.write(f"Закодована RLE послідовність: {encoded_str_rle}\n")
            out_file.write(f"Розмір закодованої RLE послідовності: {rle_size_bits} bits\n")

            # Коефіцієнт стиснення RLE
            cr_rle = round((orig_size_bits / rle_size_bits), 2) if rle_size_bits > 0 else 0
            if cr_rle < 1:
                cr_rle_display = '-'
            else:
                cr_rle_display = cr_rle
            out_file.write(f"Коефіцієнт стиснення RLE:  {cr_rle_display}\n")

            # Декодування RLE
            decoded_rle = decode_rle(encoded_rle_tuples)
            out_file.write(f"Декодована RLE послідовність:{decoded_rle}\n")
            out_file.write(f"Розмір декодованої RLE послідовності: {len(decoded_rle) * 16} bits\n\n")

            #LZW
            out_file.write("________Кодування_LZW________\n")
            encoded_lzw, lzw_size_bits = encode_zlw(sequence, out_file)

            lzw_encoded_str = "".join(map(str, encoded_lzw))
            out_file.write(f"\nЗакодована LZW послідовність:{lzw_encoded_str}\n")
            out_file.write(f"Розмір закодованої LZW послідовності: {lzw_size_bits} bits\n")

            # Коефіцієнт стиснення LZW
            cr_lzw = round((orig_size_bits / lzw_size_bits), 2) if lzw_size_bits > 0 else 0
            out_file.write(f"Коефіцієнт стиснення LZW:  {cr_lzw}\n")

            # Декодування LZW
            decoded_lzw = decode_zlw(encoded_lzw)
            out_file.write(f"Декодована LZW послідовність:{decoded_lzw}\n")
            out_file.write(f"Розмір декодованої LZW послідовності: {len(decoded_lzw) * 16} bits\n")
            out_file.write("\n")

            results_table.append([round(entropy, 2), cr_rle_display, cr_lzw])

    # 3. таблиця
    N = len(original_sequences)
    fig, ax = plt.subplots(figsize=(14 / 1.54, N / 1.54))
    headers = ['Ентропія', 'КС RLE', 'КС LZW']
    rows = [f'Послідовність {i + 1}' for i in range(N)]

    ax.axis('off')
    table = ax.table(cellText=results_table, colLabels=headers, rowLabels=rows, loc='center', cellLoc='center')
    table.set_fontsize(14)
    table.scale(0.8, 2)

    fig.savefig('Результати стиснення методами RLE та LZW.png', bbox_inches='tight')
    print("Роботу успішно завершено")


if __name__ == "__main__":
    main()