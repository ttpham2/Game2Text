import pytesseract
import subprocess
from pathlib import Path
from logger import log_text, log_media
from config import r_config, OCR_CONFIG
from util import base64_to_image, base64_to_image_path
from tools import path_to_tesseract, get_tessdata_dir, bundle_dir
from ocr_space import ocr_space_file, OCRSPACE_API_URL_USA, OCRSPACE_API_URL_EU
from google.cloud import vision

HORIZONTAL_TEXT_DETECTION = 6
VERTICAL_TEXT_DETECTON = 5

def get_temp_image_path():
    return str(Path(bundle_dir,"logs", "images", "temp.png"))

def detect_and_log(engine, cropped_image,  text_orientation, session_start_time, request_time):
    result = image_to_text(engine, cropped_image, text_orientation)
    if result is not None:
        log_text(session_start_time, request_time, result)
        log_media(session_start_time, request_time)
        return {'id': request_time, 'result': result }
    else:
        return {'error': 'OCR Failed'}

def image_to_text(engine, base64img, text_orientation):
    if engine == "OCR Space USA" or engine == "OCR Space EU":
        api_url = OCRSPACE_API_URL_USA if engine == "OCR Space USA" else OCRSPACE_API_URL_EU
        image_path = base64_to_image_path(base64img, get_temp_image_path())
        language = r_config(OCR_CONFIG, "ocr_space_language")
        return ocr_space_file(filename=image_path, language=language, url=api_url)
    elif engine == "Google Vision":
        print("Change OCR to Google Vision")
        #image = base64_to_image(base64img, get_temp_image_path())
        image_path = base64_to_image_path(base64img, get_temp_image_path())
        return google_ocr(image_path) 
    else: 
        # default to tesseract
        image = base64_to_image(base64img, get_temp_image_path())
        return tesseract_ocr(image, text_orientation)

def tesseract_ocr(image, text_orientation):
    language = r_config(OCR_CONFIG, "tesseract_language")
    psm = HORIZONTAL_TEXT_DETECTION
    # Add English Tessdata for legacy Tesseract (English is included in v4 Japanese trained data)
    is_legacy_tesseract = r_config(OCR_CONFIG, "oem") == '0'
    if is_legacy_tesseract:
        language += '+eng'
    # Manual Vertical Text Orientation
    if (text_orientation == 'vertical'):
        psm = VERTICAL_TEXT_DETECTON
        language += "_vert"
    custom_config = r'{} --oem {} --psm {} -c preserve_interword_spaces=1 {}'.format(get_tessdata_dir(), r_config(OCR_CONFIG, "oem"), psm, r_config(OCR_CONFIG, "extra_options").strip('"'))
    result = pytesseract.image_to_string(image, config=custom_config, lang=language)
    return result

tesseract_cmd = path_to_tesseract()
if tesseract_cmd is not None:
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

def google_ocr(file):
    return detect_text(file)

def detect_text(path):
    """Detects text in the file."""
    client = vision.ImageAnnotatorClient()

    with open(path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.document_text_detection(image=image)
    texts = response.text_annotations
    output = texts[0].description
    print("Texts:")

    for text in texts:
        #output = output + text.description
        print(f'\n"{text.description}"')

        vertices = [
            f"({vertex.x},{vertex.y})" for vertex in text.bounding_poly.vertices
        ]

        print("bounds: {}".format(",".join(vertices)))
    subprocess.run("clip", text=True, input=output, encoding='utf-16')
    return output
    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )