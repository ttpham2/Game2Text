import translators as ts
import requests
import time
from config import r_config, TRANSLATION_CONFIG

def multi_translate(text):
    service =  r_config(TRANSLATION_CONFIG, 'translation_service')
    if service == 'DeepL Translate':
        return deepl_translate(text)
    elif service == 'Google Translate':
        return google_translate(text)
    elif service == 'JPDB Translate':
        return jpdb_translate(text)
    else:
        return 'Error: No Translation Service Available'

def deepl_translate(text):
    text = text[:140] if len(text) > 140 else text
    response = requests.post(
    "https://www2.deepl.com/jsonrpc",
    json = {
        "jsonrpc":"2.0",
        "method": "LMT_handle_jobs",
        "params": {
            "jobs":[{
                "kind":"default",
                "raw_en_sentence": text,
                "raw_en_context_before":[],
                "raw_en_context_after":[],
                "preferred_num_beams":4,
                "quality":"fast"
            }],
            "lang":{
                "user_preferred_langs":["EN"],
                "source_lang_user_selected": r_config(TRANSLATION_CONFIG, "source_lang").upper() or "JA",
                "target_lang": r_config(TRANSLATION_CONFIG, "target_lang").upper() or "EN"
            },
            "priority":-1,
            "commonJobParams":{},
            "timestamp": int(round(time.time() * 1000))
        },
        "id": 40890008
    })
    output = response.json()
    if output is not None:
        if 'result' in output:
            return output['result']['translations'][0]['beams'][0]["postprocessed_sentence"]
        if 'error' in output:
            return 'Error: ' + output['error']['message']
    return 'Failed to Translate'

def google_translate(text):
    # ts.preaccelerate()
    result = ts.translate_text(text, 'google')
    return result

def jpdb_translate(text):
    url = "https://jpdb.io/api/v1/ja2en"
    payload = { "text": text }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": "Bearer 40d25dac559165ced0d48f2ddd2ce0db"
    }
    response = requests.post(url, json=payload, headers=headers)
    output = response.json()
    print(output)
    if output is not None:
        if 'text' in output:
            print(output['text'])
            return output['text']
        if 'error' in output:
            return 'Error: ' + output['error']['error_message']
    return 'Failed to Translate'