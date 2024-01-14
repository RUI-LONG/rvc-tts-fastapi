import requests
import winsound
import os


if __name__ == "__main__":
    api_url = ""
    # Load Model (only first time)
    model_name = "A"
    requests.post(f"{api_url}/load_model/{model_name}")

    # TTS
    data = {
        "speed": 0,
        "tts_text": "我的同事凱文，是一隻猴子",
        "tts_voice": "zh-TW-HsiaoChenNeural-Female",
    }

    response = requests.post(f"{api_url}/tts", data=data)

    if response.status_code == 200:
        wav_filename = f"downloaded.wav"

        with open(wav_filename, "wb") as wav_file:
            wav_file.write(response.content)

        if os.path.isfile(wav_filename):
            winsound.PlaySound(wav_filename, winsound.SND_FILENAME)
        else:
            print("The downloaded WAV file does not exist.")
    else:
        print("API request failed with status code:", response.status_code)
