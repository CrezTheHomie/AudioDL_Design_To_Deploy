import numpy as np
import bentoml
from bentoml.io import NumpyNdarray, File
import librosa

# Run this once to save the model
# SAVE_MODEL_PATH = "Data\model.h5"
# bentoml.keras.save_model("keyword_spotting_model",
#                          keras.models.load_model(SAVE_MODEL_PATH))

keyword_spotting_model = bentoml.keras.load_model("keyword_spotting_model")
keyword_spotting_runner = bentoml.keras.get(
    "keyword_spotting_model:v5rxcuesawl5oven").to_runner()

svc = bentoml.Service("keyword_spotting_model",
                      runners=[keyword_spotting_runner])

@svc.api(input=File(), output=NumpyNdarray())
def classify_file(input_series:bentoml.io.File) -> np.ndarray:
    signal, sr = librosa.load(input_series)

    # ensure consistency of audio file length
    if len(signal) > 22050:
        signal = signal[:22050]

    # extract MFCCs
    MFCCs = librosa.feature.mfcc(
        y=signal, n_mfcc=13, n_fft=2048, hop_length=512)
    MFCCs = MFCCs.T
    MFCCs = MFCCs[np.newaxis, ..., np.newaxis]
    return keyword_spotting_model.predict(MFCCs)

@svc.api(input=File(), output=NumpyNdarray())
def classify_file(input_series: bentoml.io.File) -> np.ndarray:
    signal, sr = librosa.load(input_series)

    # ensure consistency of audio file length
    if len(signal) > 22050:
        signal = signal[:22050]

    # extract MFCCs
    MFCCs = librosa.feature.mfcc(
        y=signal, n_mfcc=13, n_fft=2048, hop_length=512)
    MFCCs = MFCCs.T
    MFCCs = MFCCs[np.newaxis, ..., np.newaxis]
    return keyword_spotting_model.predict(MFCCs)
