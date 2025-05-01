from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_community.vectorstores import Chroma

embedding_model = LlamaCppEmbeddings(model_path='llama-2-13b.Q5_K_M.gguf')


loader = TextLoader('NDITC.txt')
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512,chunk_overlap=64)
texts = text_splitter.split_documents(docs)
db = Chroma.from_documents(texts,embedding_model,persist_directory='NDITC')


