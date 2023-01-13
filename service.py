import numpy as np
import bentoml
from bentoml.io import NumpyNdarray, File, Text
import librosa

_MAPPINGS = [
    "bed", "bird", "cat", "dog", "down", "eight",
    "five", "four", "go", "happy", "house",
    "left", "marvin", "nine", "no", "off", "on",
    "one", "right", "seven", "sheila", "six", "stop",
    "three", "tree", "two", "up", "wow", "yes", "zero"
]

# Run this once to save the model
# SAVE_MODEL_PATH = "Data\model.h5"
# bentoml.keras.save_model("keyword_spotting_model",
#                          keras.models.load_model(SAVE_MODEL_PATH))

keyword_spotting_model = bentoml.keras.load_model("keyword_spotting_model")
keyword_spotting_runner = bentoml.keras.get(
    "keyword_spotting_model:v5rxcuesawl5oven").to_runner()

svc = bentoml.Service("keyword_spotting_model",
                      runners=[keyword_spotting_runner])


# @web_static_content('./static')

@svc.api(input=File(), output=Text())
def classify_file(input_series:bentoml.io.File) -> str:
    signal, sr = librosa.load(input_series)

    # ensure consistency of audio file length
    if len(signal) > 22050:
        signal = signal[:22050]

    # extract MFCCs
    MFCCs = librosa.feature.mfcc(
        y=signal, n_mfcc=13, n_fft=2048, hop_length=512)
    MFCCs = MFCCs.T
    MFCCs = MFCCs[np.newaxis, ..., np.newaxis]
    predictions = keyword_spotting_model.predict(MFCCs)
    predicted_index = np.argmax(predictions)
    predicted_label = _MAPPINGS[predicted_index]
    return predicted_label

@svc.api(input=NumpyNdarray(), output=NumpyNdarray(dtype="int64"))
def classify_MFCCs(input_MFCCs: np.ndarray) -> np.ndarray:
    
    predictions = keyword_spotting_model.predict(input_MFCCs)
    print(f"Predictions are: {predictions}")
    
    return predictions
