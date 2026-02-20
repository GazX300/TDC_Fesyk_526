import speech_recognition as sr


def recognize_speech(recognizer, microphone):
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    result = {"текст": None}

    try:
        result["текст"] = recognizer.recognize_google(audio, show_all=False, language="uk-UA")
    except sr.UnknownValueError:
        result["текст"] = "Не вдалося розпізнати мову"
    except sr.RequestError as e:
        result["текст"] = f"Помилка сервісу: {e}"

    return result


if __name__ == "__main__":
    recognizer = sr.Recognizer()
    print("Доступні мікрофони:")
    for i, name in enumerate(sr.Microphone.list_microphone_names()):
        print(f"{i}: {name}")

    device_index = 1
    try:
        microphone = sr.Microphone(device_index=device_index)
        print(f"\nВибрано мікрофон з індексом {device_index}")
    except OSError:
        print(f"Мікрофон з індексом {device_index} не знайдено. Використовується стандартний.")
        microphone = sr.Microphone()

    print("\nСлухаю... (Для виходу натисніть Ctrl+C)")

    try:
        while True:
            result = recognize_speech(recognizer, microphone)
            if result["текст"] and result["текст"] not in ["Не вдалося розпізнати мову", "Помилка сервісу"]:
                print('Ви сказали: \n{}'.format(result["текст"]))
            else:
                print(result["текст"])
    except KeyboardInterrupt:
        print("\nПрограму завершено")