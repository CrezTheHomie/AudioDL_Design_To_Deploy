import requests
import json
from bentoml.io import File
import librosa
import numpy as np

CLASSIFY_FILE_URL = "http://127.0.0.1:3000/classify_file"
CLASSIFY_MFCCs_URL = "http://127.0.0.1:3000/classify_MFCC"
TEST_AUDIO_FILE_PATH = "Test\\00f0204f_nohash_0.wav"


def make_request_to_bento_service(
    service_url: str, input_file
) -> str:
    response = requests.post(
        service_url,
        data=input_file,
        headers={"content-type": "multipart/form-data"}
    )
    return response.text

def preprocess_file(input_file):
    signal, sr = librosa.load(input_file)

    # ensure consistency of audio file length
    if len(signal) > 22050:
            signal = signal[:22050]

    # extract MFCCs
    MFCCs = librosa.feature.mfcc(
        y=signal, n_mfcc=13, n_fft=2048, hop_length=512)
    MFCCs = MFCCs.T
    MFCCs = MFCCs[np.newaxis, ..., np.newaxis]
    return MFCCs

if __name__ == "__main__":
    audio_file = open(TEST_AUDIO_FILE_PATH, "rb")
    values = {"file": (TEST_AUDIO_FILE_PATH, audio_file, "audio/wav")}

    response = make_request_to_bento_service(
        service_url=CLASSIFY_FILE_URL, input_file=values)
    # data = response.json()

    print(f"Predicted keyword was: {response}")