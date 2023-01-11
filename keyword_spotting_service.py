import keras
import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt
MODEL_PATH = "Data\\model.h5"
NUM_SAMPLES_TO_CONSIDER = 22050

class _Keyword_Spotting_Service:
    
    model = None
    _mappings = [
        "bed", "bird", "cat", "dog", "down", "eight",
        "five", "four", "go", "happy", "house",
        "left", "marvin", "nine", "no", "off", "on",
        "one", "right", "seven", "sheila", "six", "stop",
        "three", "tree", "two", "up", "wow", "yes", "zero"
    ]
    _instance = None

    def preprocess(self, file_path, n_mfccs=13, n_fft=2048, hop_length=512):
        # load the audio file
        signal, sr = librosa.load(file_path)
        
        # ensure consistency of audio file length
        if len(signal) > NUM_SAMPLES_TO_CONSIDER:
            signal = signal[:NUM_SAMPLES_TO_CONSIDER]

        # extract MFCCs
        MFCCs = librosa.feature.mfcc(y=signal, n_mfcc=n_mfccs, n_fft=n_fft, hop_length=hop_length)
        return MFCCs.T

    def predict(self, file_path):
        # extract MFCCs
        MFCCs = self.preprocess(file_path)

        # convert 2D MFCC array into 4D array. (# samples, # segments, # coefficients, # channels = 1)
        MFCCs = MFCCs[np.newaxis,...,np.newaxis]

        # make prediction
        predictions = self.model.predict(MFCCs) # [ [range of size mappings] ]
        predicted_index = np.argmax(predictions)
        predicted_keyword = self._mappings[predicted_index]
        
        return predicted_keyword

# force singleton class
def Keyword_Spotting_Service():
    # ensure we only have one instance of KSS
    if _Keyword_Spotting_Service._instance is None:
        _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
        _Keyword_Spotting_Service.model = keras.models.load_model(MODEL_PATH)
    return _Keyword_Spotting_Service._instance

if __name__ == "__main__":

    kss = Keyword_Spotting_Service()
    keyword1 = kss.predict("Test\\00f0204f_nohash_0.wav")
    keyword2 = kss.predict("Test\\0ac15fe9_nohash_0.wav")
    keyword3 = kss.predict("Test\\1a9afd33_nohash_1.wav")

    print(f"Predicted keywords:\n1 {keyword1}\n2 {keyword2}\n3 {keyword3}")
