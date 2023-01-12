import requests

CLASSIFY_FILE_URL = "http://127.0.0.1:3000/classify_file"
CLASSIFY_MFCCs_URL = "http://127.0.0.1:3000/classify_MFCCs"
TEST_AUDIO_FILE_PATH = "Test\\0ac15fe9_nohash_0.wav"


def make_multipart_request_to_bento_service(
    service_url: str, input_files: list
) -> str:
    print("Responses starting")
    pass_files = {}
    i = 0
    for file in input_files:
        print(f"I see file: {file}\n")
        pass_files['file' + str(i)] = ('TestFile' + str(i), open(file, 'rb'),
                                       'application/octet-stream', {'Expires': '0'})
        i += 1
    print(f"Sending {pass_files}")
    response = requests.post(
        service_url, files=pass_files)
    return response.text

def make_request_to_bento_service(
    service_url: str, input_file, header_dict: dict
) -> str:

    response = requests.post(
        service_url, headers=header_dict, data=input_file)
    return response.text

if __name__ == "__main__":
    
    with open(TEST_AUDIO_FILE_PATH, 'rb') as f:
        audio_data = f.read()

    headers_dict = {
        'accept': 'application/json',
        'Content-Type': 'application/octet-stream',
    }

    response = make_request_to_bento_service(
        service_url=CLASSIFY_FILE_URL, input_file=audio_data, header_dict=headers_dict)
    # files_list = ["Test\\00f0204f_nohash_0.wav",
    #               "Test\\0ac15fe9_nohash_0.wav"]
    # response = make_multipart_request_to_bento_service(
    #     CLASSIFY_FILE_URL, files_list)

    print(f"Response was: {response}")
