import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 1. Cargar todos los archivos PDF
documents = []
directory = "documentos"
for filename in os.listdir(directory):
    if filename.endswith(".pdf"):
        path = os.path.join(directory, filename)
        loader = PyPDFLoader(path)
        documents.extend(loader.load())

# 2. Fragmentar los documentos
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# 3. Embeddings
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# 4. Vectorstore persistente
vectordb = Chroma.from_documents(
    docs,
    embedding_function,
    persist_directory="./sqlite_db"
)

# 5. Modelo LLM
llm = OllamaLLM(model="llama3")

# 6. Prompt en español
prompt_es = PromptTemplate(
    template="Contesta en español de forma clara y concisa:\n\nContexto:\n{context}\n\nPregunta:\n{question}",
    input_variables=["context", "question"]
)

# 7. Cadena RAG con prompt
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(search_kwargs={"k": 5}),
    chain_type_kwargs={"prompt": prompt_es},
    return_source_documents=True
)

# 8. Consulta
query = "Técnicas Básicas de Conteo"
result = qa_chain.invoke(query)

# 9. Mostrar respuesta
print("\n Respuesta:")
print(result["result"])

# 10. Mostrar documentos fuente con score
print("\n📚 Documentos fuente con puntaje de similitud:")
results_with_score = vectordb.similarity_search_with_score(query, k=5)
vistas = set()
for doc, score in results_with_score:
    fuente = doc.metadata.get("source", "Desconocido")
    if fuente not in vistas:
        print(f"- {fuente} → Score: {score:.4f}")
        vistas.add(fuente)