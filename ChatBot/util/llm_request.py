import requests
import json


def request_llm(prompt, history, images_str=None, **gen_kwargs):
    base_url = 'http://127.0.0.1:25'
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url=base_url, headers=headers, json={"prompt": prompt, "history": history, "images_str":images_str, **gen_kwargs})
    result = json.loads(response.text)
    assert result['status'] == 200
    return result


# if __name__ == '__main__':
#     response = request_llm('你猜', [])
#     print(response)
