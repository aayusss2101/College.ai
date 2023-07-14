from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub, OpenAI

# from PyPDF2 import PdfReader

# reader=PdfReader('time table.pdf')

# for i, page in enumerate(reader.pages):
#     print(i)
#     text=page.extract_text()
#     print(text)
#     print()

import os

os.environ["HUGGINGFACEHUB_API_TOKEN"]="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"


loader=TextLoader('state_of_the_union.txt')
documents=loader.load()
text_splitter=CharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
texts=text_splitter.split_documents(documents)
model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings=HuggingFaceEmbeddings(model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs)
docsearch=Chroma.from_documents(texts,embeddings)
qa = RetrievalQA.from_chain_type(llm=HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.5, "max_length": 64}), chain_type="stuff", retriever=docsearch.as_retriever(),return_source_documents=True)
query = "What did the president say about Ketanji Brown Jackson"
result=qa({"query": query})
print(result["result"])
# print(result["source_documents"])
