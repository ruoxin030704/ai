# 參考chatgpt
import openai
import os
import faiss
import numpy as np
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings  # 修改引用路徑
from langchain.llms import OpenAI

# 設置 OpenAI API 密鑰
openai.api_key = os.getenv('my key')

# 示例文檔
documents = [
    "LangChain 是一個用於構建 LLM 驅動應用的框架。",
    "RAG 是一種結合檢索和生成的技術，用於回答用戶問題。",
    "ReAct 是一種基於檢索增強的推理策略。"
]

# 創建向量嵌入
embeddings = OpenAIEmbeddings()
document_embeddings = embeddings.embed_documents(documents)

# 使用 FAISS 創建向量存儲
dimension = len(document_embeddings[0])
index = faiss.IndexFlatL2(dimension)
index.add(np.array(document_embeddings))

class FAISSVectorStore:
    def __init__(self, index, documents):
        self.index = index
        self.documents = documents

    def retrieve(self, query, k=5):
        query_embedding = embeddings.embed_query(query)
        D, I = self.index.search(np.array([query_embedding]), k)
        return [self.documents[i] for i in I[0]]

vector_store = FAISSVectorStore(index, documents)

# 創建檢索增強的生成模型
qa_chain = RetrievalQA(llm=OpenAI(), retriever=vector_store.retrieve)

def main():
    print("歡迎來到 RAG 系統！輸入 'exit' 以退出。")
    while True:
        user_input = input("你: ")
        if user_input.lower() == 'exit':
            break
        response = qa_chain.run(input_text=user_input)
        print(f"RAG: {response}")

if __name__ == "__main__":
    main()
