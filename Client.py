import requests

CLASSIFY_FILE_URL = "http://127.0.0.1:3000/classify_file"
CLASSIFY_MFCCs_URL = "http://127.0.0.1:3000/classify_MFCCs"
TEST_AUDIO_FILE_PATH = "Test\\00f0204f_nohash_0.wav"
RAND_PDF_PATH = "Test\\ml-canvas.pdf"


def make_request_to_bento_service(
    service_url: str, input_file, header_dict: dict
) -> str:

    # with open(TEST_AUDIO_FILE_PATH, 'rb') as f:
    #     audio_data = f.read()

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
    
    print(f"Response was: {response}")
