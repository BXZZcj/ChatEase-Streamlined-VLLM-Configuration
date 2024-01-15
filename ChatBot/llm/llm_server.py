import sys
import os
# 测试修改
# 获取当前脚本文件的绝对路径
script_path = os.path.abspath(__file__)
# 获取当前脚本的父目录的父目录（即 config 和 llm 所在的目录）
project_dir = os.path.dirname(os.path.dirname(script_path))
# 将 project_dir 添加到 sys.path
if project_dir not in sys.path:
    sys.path.append(project_dir)


import uvicorn
import json
import datetime
import requests
from PIL import Image
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from transformers import LlavaForConditionalGeneration, AutoModelForCausalLM,AutoTokenizer, AutoProcessor
from config.model_config import *
from typing import List, Tuple
import io
import base64


DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/_stcore/health")
async def health_check():
    return {"message": "healthy"}


@app.get("/_stcore/allowed-message-origins")
async def allowed_message_origins_check():
    return {"message": "allowed"}


@app.post("/")
async def create_item(request: Request):
    global model, tokenizer
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    prompt = json_post_list.get('prompt')
    history = json_post_list.get('history')
    max_length = json_post_list.get('max_length')
    top_p = json_post_list.get('top_p')
    # top_k = json_post_list.get('top_k')
    temperature = json_post_list.get('temperature')
    print(prompt)
    response, history = chat_with_history(query=prompt,
                                          history=history,
                                          max_length=max_length if max_length else 2048,
                                          top_p=top_p if top_p else 0.7,
                                          # top_k=top_k if top_k else 1,
                                          do_sample=True,
                                          temperature=temperature if temperature else 1e-8)
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response,
        "history": history,
        "status": 200,
        "time": time
    }
    log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'
    print(log)
    torch_gc()
    return answer


@torch.inference_mode()
def chat_with_history(query: str, history: List[Tuple[str, str]] = None, **gen_kwargs):
    """
    带有历史信息的对话.
    :param query: 用户输入的对话语句.
    :param history: 历史对话信息, (query, response)组成的List.
    :param gen_kwargs: 生成参数, 参考https://huggingface.co/docs/transformers/v4.33.2/en/main_classes/text_generation#transformers.GenerationConfig.
    :return: response: str, history: List[Tuple[str, str]]
    """
    global model, tokenizer
    if history is None:
        history = []
    prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, " \
             "detailed, and polite answers to the user's questions.\n\n"
    for (old_query, response) in history:
        prompt += f"USER: {old_query}\nASSISTANT: {response}</s>\n"
    prompt += f"USER: {query}\nASSISTANT: "
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()
    output = model.generate(inputs=input_ids, **gen_kwargs)
    output = output.tolist()[0][len(input_ids[0]):]
    response = tokenizer.decode(output)
    history = history + [(query, response)]
    return response, history


# # 定义一个 POST 请求处理函数
# @app.post("/")
# async def create_item(request: Request):
#     global model, processor

#     # 从请求中获取 JSON 数据
#     json_post_raw = await request.json()
#     json_post = json.dumps(json_post_raw)
#     json_post_list = json.loads(json_post)

#     # 从 JSON 数据中提取所需的参数
#     prompt = json_post_list.get('prompt')  # 提取提示信息
#     history = json_post_list.get('history')  # 提取对话历史
#     max_length = json_post_list.get('max_length', 20408)  # 提取生成文本的最大长度，默认为 20408
#     top_p = json_post_list.get('top_p', 0.7)  # 提取 top_p 参数，默认为 0.7
#     temperature = json_post_list.get('Temperature', 1e-8)  # 提取温度参数，默认为 1e-8
#     images_str = json_post_raw.get('images_str')  # 提取图像数据的 Base64 字符串

#     images=None
#     if images_str:
#         # 解码图像数据
#         images_bytes = base64.b64decode(images_str.encode('utf-8'))
#         images = Image.open(io.BytesIO(images_bytes))
#     else:
#         # images = Image.new('RGB', (14, 14), color = 'white')
#         pass

#     # 调用 chat_with_history 函数进行文本生成
#     response, history = chat_with_history(
#         query=prompt,
#         history=history,
#         max_length=max_length,
#         top_p=top_p,
#         temperature=temperature,
#         images=images,  # 将图像作为参数传递给 chat_with_history 函数
#         do_sample=True
#     )

#     # 获取当前时间
#     now = datetime.datetime.now()
#     time = now.strftime("%Y-%m-%d %H:%M:%S")

#     # 构造响应数据
#     answer = {
#         "response": response,
#         "history": history,
#         "status": 200,
#         "time": time
#     }

#     # 构造日志信息
#     log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'

#     # 打印日志信息
#     print(log)

#     # 执行 torch 垃圾收集
#     torch_gc()

#     # 返回响应数据
#     return answer


# @torch.inference_mode()
# def chat_with_history(query: str, history: List[Tuple[str, str]] = None, images=None, **gen_kwargs):
#     global model, processor, tokenizer
#     if history is None:
#         history = []
#     prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, " \
#              "detailed, and polite answers to the user's questions.\n\n"
#     for (old_query, response) in history:
#         prompt += f"USER: {old_query}\nASSISTANT: {response}</s>\n"
#     prompt += f"USER: {query}\nASSISTANT: "
#     if images:
#         prompt="<image>\n"+prompt
    
#     inputs = processor(text=prompt, images=images, return_tensors="pt").input_ids.cuda()
#     inputs.to('cuda')
#     generate_ids = model.generate(inputs=inputs, **gen_kwargs)
#     response = processor.batch_decode(generate_ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]

#     # input_ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()
#     # output = model.generate(inputs=input_ids, **gen_kwargs)
#     # output = output.tolist()[0][len(input_ids[0]):]
#     # response = tokenizer.decode(output)

#     print(response)
#     history.append((query, response))
#     return response, history


if __name__ == '__main__':
    model_path = llm_model_dict['vicuna-13b-v1.5-GPTQ']['local_model_path']
    # model = LlavaForConditionalGeneration.from_pretrained(model_path, device_map="auto")
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    # processor= AutoProcessor.from_pretrained(model_path)
    tokenizer=AutoTokenizer.from_pretrained(model_path)
    model.eval()    
    uvicorn.run(app, host='0.0.0.0', port=25, workers=1)
    