import requests
import json

URL = "http://127.0.0.1:3000/predict"
TEST_AUDIO_FILE_PATH = "Test\\00f0204f_nohash_0.wav"


def make_request_to_bento_service(
    service_url: str, input_file
) -> str:
    response = requests.post(
        service_url,
        data=input_file,
        headers={"content-type": "application/json"}
    )
    return response.text

if __name__ == "__main__":
    audio_file = open(TEST_AUDIO_FILE_PATH, "rb")
    values = {"file": (TEST_AUDIO_FILE_PATH, audio_file, "audio/wav")}

    response = make_request_to_bento_service(service_url=URL, input_file=values)
    # data = response.json()

    print(f"Predicted keyword was: {response}")