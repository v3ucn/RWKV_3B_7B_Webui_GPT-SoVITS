import gradio as gr
import os, gc, copy, torch
from datetime import datetime
from pynvml import *

import requests
import json



# Flag to check if GPU is present
HAS_GPU = True

# Model title and context size limit
ctx_limit = 2000
title = "RWKV-5-H-World"
model_file = "rwkv-5-h-world-7B"

from pydub import AudioSegment
from pydub.playback import play

os.environ['RWKV_JIT_ON'] = '1'
os.environ["RWKV_CUDA_ON"] = '1'
# MODEL_STRAT = "cuda bf16"
# MODEL_STRAT = "cuda fp16i8"

from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS

pipeline = None
model = None

# Get the GPU count
try:
    nvmlInit()
    GPU_COUNT = nvmlDeviceGetCount()
    if GPU_COUNT > 0:
        HAS_GPU = True
        gpu_h = nvmlDeviceGetHandleByIndex(0)
except NVMLError as error:
    print(error)


def read_now(text,api_url):

    if text == "":
        return "请输入要播放的文本"

    data = json.dumps({"text":text})#转换为json字符串
    headers = {"Content-Type":"application/json"}#指定提交的是json
    r = requests.post(f"{api_url}",data=data,headers=headers)

    with open('./output_audio.wav', 'wb') as audio_file:
        audio_file.write(r.content)
    sound = AudioSegment.from_file("./output_audio.wav", format="wav")
    play(sound)


def load_model(model_name,MODEL_STRAT):

    # Load the model accordingly

    global model,pipeline
    
    model = RWKV(model=model_name, strategy=MODEL_STRAT)
    
    pipeline = PIPELINE(model,"rwkv_vocab_v20230424")

    return "加载成功"


# Prompt generation
def generate_prompt(instruction, input=""):
    instruction = instruction.strip().replace('\r\n','\n').replace('\n\n','\n')
    input = input.strip().replace('\r\n','\n').replace('\n\n','\n')
    if input:
        return f"""Instruction: {instruction}
Input: {input}
Response:"""
    else:
        return f"""User: hi
Assistant: Hi. I am your assistant and I will provide expert full response in full details. Please feel free to ask any question and I will always answer it.
User: {instruction}
Assistant:"""

# Evaluation logic
def evaluate(
    ctx,
    token_count=200,
    temperature=1.0,
    top_p=0.7,
    presencePenalty = 0.1,
    countPenalty = 0.1,
):
    print(ctx)
    args = PIPELINE_ARGS(temperature = max(0.2, float(temperature)), top_p = float(top_p),
                     alpha_frequency = countPenalty,
                     alpha_presence = presencePenalty,
                     token_ban = [], # ban the generation of some tokens
                     token_stop = [0]) # stop generation whenever you see any token here
    ctx = ctx.strip()
    all_tokens = []
    out_last = 0
    out_str = ''
    occurrence = {}
    state = None
    for i in range(int(token_count)):
        out, state = model.forward(pipeline.encode(ctx)[-ctx_limit:] if i == 0 else [token], state)
        for n in occurrence:
            out[n] -= (args.alpha_presence + occurrence[n] * args.alpha_frequency)

        token = pipeline.sample_logits(out, temperature=args.temperature, top_p=args.top_p)
        if token in args.token_stop:
            break
        all_tokens += [token]
        for xxx in occurrence:
            occurrence[xxx] *= 0.996
        if token not in occurrence:
            occurrence[token] = 1
        else:
            occurrence[token] += 1

        tmp = pipeline.decode(all_tokens[out_last:])
        if '\ufffd' not in tmp:
            out_str += tmp
            yield out_str.strip()
            out_last = i + 1

    if HAS_GPU == True :
        gpu_info = nvmlDeviceGetMemoryInfo(gpu_h)
        print(f'vram {gpu_info.total} used {gpu_info.used} free {gpu_info.free}')

    del out
    del state
    gc.collect()

    if HAS_GPU == True :
        torch.cuda.empty_cache()

    yield out_str.strip()

def main():

    # Gradio blocks
    with gr.Blocks(title=title) as demo:
        gr.HTML(f"<div style=\"text-align: center;\">\n<h1>RWKV-5 World v2 - {title}</h1>\n</div>")

        gr.HTML(f"<div style=\"text-align: center;\">\n<h1>加载方式说明</h1>\nfp16:这是一种16位浮点数精度，也称为半精度。它在GPU上表现良好，因为它可以减少内存占用并提高计算速度。但是，它不支持CPU，因为大多数CPU不支持半精度计算。\nfp16i8:结合了半精度或单精度浮点数与8位整数量化。量化可以显著减少模型所需的内存（VRAM对于GPU，RAM对于CPU），但可能会牺牲一些精度，并且由于量化和反量化的过程，可能会导致计算速度变慢。\n cuda fp16i8 *10 -> cpu fp32'：这种策略首先在GPU上使用fp16i8精度运行模型的前10层，然后将模型转移到CPU上，并使用fp32精度继续运行。这种混合策略可以在利用GPU的并行计算能力的同时，减少内存消耗，并在CPU上完成剩余的计算。</div>")

            
        with gr.Tab("RWKV-5-h"):
            gr.Markdown(f"This is RWKV-5-h ")

            with gr.Row():
                with gr.Column():
                    model_dropdown = gr.Dropdown(label="模型列表", choices=["./rwkv-5-h-world-3B.pth", "./rwkv-5-h-world-7B.pth"], value="./rwkv-5-h-world-3B.pth", interactive=True)
                    start_dropdown = gr.Dropdown(label="加载方式", choices=["cuda fp16", "cuda fp16i8","cuda fp16i8 *10 -> cpu fp32"], value="cuda fp16", interactive=True)

                    b_load = gr.Button("加载模型", variant="primary")
                    b_output = gr.Textbox(label="加载结果")

            b_load.click(load_model, [model_dropdown, start_dropdown], [b_output])


            with gr.Row():
                with gr.Column():
                    prompt = gr.Textbox(lines=2, label="Prompt", value="""边儿上还有两条腿，修长、结实，光滑得出奇，潜伏着
    媚人的活力。他紧张得脊梁都皱了起来。但他不动声色。""")
                    token_count = gr.Slider(10, 500, label="Max Tokens", step=10, value=200)
                    temperature = gr.Slider(0.2, 2.0, label="Temperature", step=0.1, value=1.0)
                    top_p = gr.Slider(0.0, 1.0, label="Top P", step=0.05, value=0.3)
                    presence_penalty = gr.Slider(0.0, 1.0, label="Presence Penalty", step=0.1, value=1)
                    count_penalty = gr.Slider(0.0, 1.0, label="Count Penalty", step=0.1, value=1)
                with gr.Column():
                    with gr.Row():
                        submit = gr.Button("开始推理", variant="primary")
                        clear = gr.Button("Clear", variant="secondary")
                    output = gr.Textbox(label="Output", lines=5)
                    read_b = gr.Button("开始朗读", variant="primary")
            data = gr.Dataset(components=[prompt, token_count, temperature, top_p, presence_penalty, count_penalty], label="Example Instructions", headers=["Prompt", "Max Tokens", "Temperature", "Top P", "Presence Penalty", "Count Penalty"])
            submit.click(evaluate, [prompt, token_count, temperature, top_p, presence_penalty, count_penalty], [output])
            clear.click(lambda: None, [], [output])
            data.click(lambda x: x, [data], [prompt, token_count, temperature, top_p, presence_penalty, count_penalty])
            api_url = gr.Textbox(label="GPT-SoVITS接口地址", value="http://localhost:9880/tts_to_audio/")
            read_b.click(read_now,[output,api_url],[])

    demo.queue()
    demo.launch(server_name="0.0.0.0",inbrowser=True)

if __name__ == '__main__':
    main()
