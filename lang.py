from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.embeddings import LlamaCppEmbeddings



llm = LlamaCpp(
    model_path="llama-2-13b.Q5_K_M.gguf",
    temperature=0.75,
    n_gpu_layers=0,
    n_ctx=2048,
    max_tokens=2000,
    verbose=True, 
)
embedding_model = LlamaCppEmbeddings(model_path='llama-2-13b.Q5_K_M.gguf')


db = Chroma(persist_directory='NDITC', embedding_function=embedding_model)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever= db.as_retriever(),
)
def generate_text(x):
    print(qa_chain.run(x))

while True:
    generate_text(input("ASK ANYTHING : "))
