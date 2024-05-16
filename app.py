from flask import Flask, render_template, request, jsonify

# Import your custom NLP modules and setup code
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA

app = Flask(__name__)

directory = 'data'

def load_docs(directory):
  loader = DirectoryLoader(directory)
  documents = loader.load()
  return documents

documents = load_docs(directory)
len(documents)

from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_docs(documents,chunk_size=1000,chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs

docs = split_docs(documents)
print(len(docs))

from langchain.embeddings import SentenceTransformerEmbeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

from langchain.vectorstores import Chroma
db = Chroma.from_documents(docs, embeddings)

persist_directory = "chroma_db"

vectordb = Chroma.from_documents(
    documents=docs, embedding=embeddings, persist_directory=persist_directory
)
vectordb.persist()
new_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

chatgptapikey=" paste api key here "

import os
os.environ["OPENAI_API_KEY"] = chatgptapikey

from langchain.chat_models import ChatOpenAI
model_name = "gpt-3.5-turbo"
llm = ChatOpenAI(model_name=model_name)


from langchain.chains.question_answering import load_qa_chain
chain = load_qa_chain(llm, chain_type="stuff",verbose=True)



@app.route('/')
def home():
    return render_template('index.html')

 
@app.route('/chat', methods=['POST'])
def chat():

    query = request.form['message']

    matching_docs = db.similarity_search(query)
    print("=====================================================")
    answer =  chain.run(input_documents=matching_docs, question=query)
    
    from langchain.chains import RetrievalQA
    retrieval_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=db.as_retriever())
    retrieval_chain.run(query)

    return jsonify({'message': answer})

if __name__ == '__main__':
    app.run(debug=True)
