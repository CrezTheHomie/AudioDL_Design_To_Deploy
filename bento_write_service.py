import numpy as np
import bentoml
from bentoml.io import NumpyNdarray

# Run this once to save the model
# SAVE_MODEL_PATH = "Data\model.h5"
# bentoml.keras.save_model("keyword_spotting_model",
#                          keras.models.load_model(SAVE_MODEL_PATH))

keyword_spotting_model = bentoml.keras.load_model("keyword_spotting_model")

svc = bentoml.Service("kss", runners=keyword_spotting_model)

@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(input_series:np.ndarray) -> np.ndarray:
    return keyword_spotting_model.predict(input_series)