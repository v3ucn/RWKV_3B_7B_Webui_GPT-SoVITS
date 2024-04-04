"""
查询文本
"""
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings()
query_text = "边儿上还有两条腿，修长、结实，光滑得出奇，潜伏着媚人的活力。"

"""
向量数据库查询
"""
from langchain.vectorstores import FAISS

db = FAISS.load_local('./db', embeddings,allow_dangerous_deserialization=True)
docs = db.similarity_search(query_text)
doc_strs = []
for doc in docs:
    doc_strs.append(doc.page_content)
doc_strs = '\n'.join(doc_strs)

"""
构建提示词
"""

print(doc_strs)

from langchain.prompts import PromptTemplate

prompt = PromptTemplate.from_template("Instruction: {query}\n\nInput: {doc}\n\nResponse:")
prompt = prompt.format(doc=doc_strs, query=query_text)
# print(prompt)


# """
# 加载模型
# """
# import os, sys

# current_path = os.path.dirname(os.path.abspath(__file__))


# import numpy as np
# np.set_printoptions(precision=4, suppress=True, linewidth=200)

# os.environ["RWKV_JIT_ON"] = '1'
# os.environ["RWKV_CUDA_ON"] = '1'

# MODEL_NAME = './rwkv-5-h-world-3B.pth'

# import torch
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.allow_tf32 = True
# torch.backends.cuda.matmul.allow_tf32 = True

# from rwkv.model import RWKV
# from rwkv.utils import PIPELINE

# model = RWKV(model=MODEL_NAME, strategy='cuda fp16i8')
# pipeline = PIPELINE(model, "rwkv_vocab_v20230424")

# PAD_TOKENS = []
# out_tokens = []
# out_last = 0
# out_str = ''
# occurrence = {}
# state = None
# ctx = prompt
# for i in range(8000):
#     tokens = PAD_TOKENS + pipeline.encode(ctx) if i == 0 else [token]

#     out, state = pipeline.model.forward(tokens, state)
#     for n in occurrence:
#         out[n] -= (0.4 + occurrence[n] * 0.4)  # repetition penalty

#     token = pipeline.sample_logits(out, temperature=1, top_p=0.2)
#     if token == 0:
#         break  # exit when 'endoftext'

#     out_tokens += [token]
#     occurrence[token] = 1 + (occurrence[token] if token in occurrence else 0)

#     tmp = pipeline.decode(out_tokens[out_last:])
#     if ('\ufffd' not in tmp) and (
#             not tmp.endswith('\n')):  # only print when the string is valid utf-8 and not end with \n
#         print(tmp, end='', flush=True)
#         out_str += tmp
#         out_last = i + 1

#     if '\n\n' in tmp:  # exit when '\n\n'
#         out_str += tmp
#         out_str = out_str.strip()
#         break