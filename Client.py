import requests
import json

URL = "http://127.0.0.1:5000/predict"
TEST_AUDIO_FILE_PATH = "Test\\00f0204f_nohash_0.wav"


def make_request_to_bento_service(
    service_url: str, input_file
) -> str:
    serialized_input_data = json.dumps(input_file.tolist())
    response = requests.post(
        service_url,
        data=serialized_input_data,
        headers={"content-type": "application/json"}
    )
    return response.text

if __name__ == "__main__":
    audio_file = open(TEST_AUDIO_FILE_PATH, "rb")
    values = {"file": (TEST_AUDIO_FILE_PATH, audio_file, "audio/wav")}
    response = requests.post(url=URL, files=values)
    data = response.json()

    print(f"Predicted keyword was: {data['keyword']}")