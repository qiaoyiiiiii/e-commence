# e:/RAG/src/langchain_rag/advanced_langchain_rag.py
import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ========== 1. 加载文档 ==========
# 假设 docs 目录在项目根目录下，调整路径
current_dir = os.path.dirname(os.path.abspath(__file__))
docs_dir = os.path.join(current_dir, "../../data/kb") # 使用 data/kb 作为文档源

# 由于您给的示例是 PyPDFLoader，但我们的kb是md文件，这里调整为加载Markdown
# 如果您确实有PDF文件，需要将data/kb下的md文件换成pdf文件，或者调整loader
loader = DirectoryLoader(docs_dir, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader)
documents = loader.load()

# ========== 2. 切片 ==========
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", " ", ""]
)
chunks = splitter.split_documents(documents)

# ========== 3. Embedding 模型 ==========
# 请确保您的设备支持CUDA，如果不支持，可以尝试将device改为"cpu"或移除
embedding_model = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-large-zh-v1.5",
    model_kwargs={"device": "cpu"}, # 注意：如果无CUDA设备，请改为 "cpu"
    encode_kwargs={"normalize_embeddings": True},
    query_instruction="为这个句子生成表示以用于检索相关文章："
)

# ========== 4. 向量数据库 ==========
# 首次建库
chroma_db_path = os.path.join(current_dir, "../../chroma_db") # 调整持久化路径
if not os.path.exists(chroma_db_path):
    print(f"Chroma DB not found at {chroma_db_path}, creating new one...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=chroma_db_path,
        collection_name="company_docs"
    )
    vectorstore.persist()
    print("Chroma DB created and persisted.")
else:
    # 已有库直接加载
    print(f"Loading existing Chroma DB from {chroma_db_path}...")
    vectorstore = Chroma(
        persist_directory=chroma_db_path,
        embedding_function=embedding_model,
        collection_name="company_docs"
    )
    print("Chroma DB loaded.")

# ========== 5. Reranker ==========
# 请确保您的设备支持CUDA，如果不支持，可以尝试将device改为"cpu"或移除
reranker = CrossEncoderReranker(
    model=HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-large", model_kwargs={"device": "cpu"}), # 注意：如果无CUDA设备，请改为 "cpu"
    top_n=5
)
retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 20}) # 先检索20个，再重排
)

# ========== 6. Prompt ==========
prompt = PromptTemplate(
    template="""严格根据以下文档回答问题，找不到信息就说找不到。 

参考文档：
{context}

问题：{question}
回答：""",
    input_variables=["context", "question"]
)

# ========== 7. 组装 RAG 链 ==========
# 请确保 Ollama 服务正在运行，并且模型已拉取
# 可以从 .env 文件读取Ollama配置
from dotenv import load_dotenv
load_dotenv()
ollama_model_name = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
ollama_host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")

llm = Ollama(model=ollama_model_name, temperature=0, base_url=ollama_host)

def format_docs(docs):
    return "\n\n".join([
        f"[来源:{d.metadata.get('source','').split(os.sep)[-1]}]\n{d.page_content}" # 提取文件名作为来源
        for d in docs
    ])

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)

# ========== 8. 使用 ==========
def run_advanced_rag(question: str):
    print("\n" + "=" * 70)
    print(f"问题: {question}")
    print("-" * 70)
    print("回答:")
    response_chunks = []
    for chunk in rag_chain.stream(question):
        print(chunk, end="", flush=True)
        response_chunks.append(chunk)
    print("\n" + "=" * 70)
    return "".join(response_chunks)

if __name__ == "__main__":
    # 命令行参数解析
    import argparse
    parser = argparse.ArgumentParser(description="Advanced Langchain RAG Demo")
    parser.add_argument("--question", required=True, help="Question for the RAG system")
    args = parser.parse_args()

    run_advanced_rag(args.question)
