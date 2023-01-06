import requests

URL = "http://127.0.0.1:5000/predict"
TEST_AUDIO_FILE_PATH = "Test\\00f0204f_nohash_0.wav"

if __name__ == "__main__":
    audio_file = open(TEST_AUDIO_FILE_PATH, "rb")
    values = {"file": (TEST_AUDIO_FILE_PATH, audio_file, "audio/wav")}
    response = requests.post(url=URL, files=values)
    data = response.json()

    print(f"Predicted keyword was: {data['keyword']}")