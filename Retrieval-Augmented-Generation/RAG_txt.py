from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM

# 1. Cargar documento
loader = TextLoader("documentos/ganaderia_precision.txt")
documents = loader.load()

# 2. Fragmentar
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# 3. Embeddings actualizados
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. Vectorstore (Chroma usa SQLite por defecto)
vectordb = Chroma.from_documents(
    docs,
    embedding_function,
    persist_directory="./sqlite_db"
)

# 5. LLM actualizado con langchain-ollama
llm = OllamaLLM(model="llama3")

# 6. Cadena QA
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True
)

# 7. Consulta
query = "¿Qué dice el texto sobre la ganadería de precisión? en español"
result = qa_chain.invoke(query)

# 8. Mostrar resultados
print("\n Respuesta:")
print(result["result"])

print("\n Documentos fuente:")
for doc in result["source_documents"]:
    print(doc.metadata)