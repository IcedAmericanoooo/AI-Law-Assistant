# 简单RAG的几个步骤
'''
# 第一行代码：导入相关的库
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader 
# 第二行代码：加载数据
documents = SimpleDirectoryReader(input_files=["law_knowleadge.txt"]).load_data() 
# 第三行代码：构建索引
index = VectorStoreIndex.from_documents(documents)
# 第四行代码：创建问答引擎
query_engine = index.as_query_engine()
# 第五行代码: 开始问答
print(query_engine.query("寻衅滋事?"))
'''

# 高级RAG的步骤
'''
1. 导入数据
2. 文本切块
3. 向量嵌入
4. 向量存储
5. 检索生成
'''

# 利用langchain实现的RAG
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import ModelScopeEmbeddings
from langchain_community.vectorstores import FAISS


# 构造提示词模板
def langchain_RAG(data_path):
    # 加载数据
    loader = CSVLoader(
        file_path = data_path,
        csv_args={
            "delimiter": ",",
            "quotechar": "",
            "fieldnames":["Query", "Answer"]
        }
    )
    data = loader.load()
    # 加载embedding模型，用于将数据进行chunk向量化，我这里使用的是gte_Qwen2-7B-instruct
    embeddings = ModelScopeEmbeddings(model_id = 'iic/gte_Qwen2-7B-instruct')

    # chunk之后的模型存入faiss本地向量数据库
    vector_db = FAISS.from_documents(data, embeddings)
    vector_db.save_local('law_assitant.faiss')
    print('faiss saved')



import pandas as pd
from dual_model import DualModel
from transformers import AutoTokenizer
import torch
from tqdm import tqdm
import faiss
# 检索答案
def self_RAG(data_path):
    # 读入数据
    data = pd.read_csv("./law_faq.csv")
    # 加载模型
    dual_model = DualModel.from_pretrained("../pretraied_models_checkpoint-500")  # 基于bert训练的一个模型
    dual_model.cuda()
    dual_model.eval()

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-macbert-base')

    # 把title列都读进来，并且构造成向量
    questions = data['title'].to_list()
    vectors = []
    with torch.inference_mode():
        for i in tqdm(range(0, len(questions), 32)):  # 32个元素为一个batch
            batch_sens = questions[i, i + 32]
            inputs = tokenizer(batch_sens, return_tensors = 'pt', padding = True, max_length = 128, truncation = True)
            inputs = {k: v.to(dual_model.device)for k, v in inputs.items()}
            vector = dual_model.bert(**inputs)[1]
            vectors.append(vector)
    vectors = torch.concat(vectors, dim = 0).cpu().numpy()

    # 创建索引
    index = faiss.IndexFlatIP(768)
    faiss.normalize_L2(vectors)
    index.add(vectors)

    # 保存到本地
    vector_db = FAISS.from_embeddings(embeddings=index, embedding=dual_model)
    vector_db.save_local('law_assitant.faiss')
    print('faiss saved')
