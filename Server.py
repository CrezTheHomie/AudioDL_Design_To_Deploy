from flask import Flask, request, jsonify
import random
import os
from keyword_spotting_service import Keyword_Spotting_Service

"""
server will listen to port 5000

client -> POST request -> server -> prediction back to client
"""

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    # unpack the audio file and save
    audio_file = request.files["file"]
    file_name = str(random.randint(0, 1000))
    audio_file.save(file_name)

    # invoke the kss
    kss = Keyword_Spotting_Service()

    # make a prediction
    predicted_keyword = kss.predict(file_name)

    # remove audio file
    os.remove(file_name)

    # send back the predicted keyword as JSON
    data = {"keyword": predicted_keyword}

    return jsonify(data)


if __name__ == "__main__":
    app.run(debug=True)
