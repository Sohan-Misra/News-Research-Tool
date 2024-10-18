import os
import getpass
import streamlit as st
import time
import langchain
from dotenv import load_dotenv
#from langchain.chains import RetrievalQAWithSourcesChain
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
#from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter yout Api key")

llm = ChatGoogleGenerativeAI(
    model = "gemini-1.5-flash",
    temperature=0.9
)

st.title("News Research Tool")
st.sidebar.title("News Article URLs")
filename = "faiss_store_google"
urls = []
for i in range(2):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_pressed = st.sidebar.button("PROCESS URLs")

main_placeholder = st.empty()

if process_url_pressed:
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Loading content from URLs...")
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        separators = ['\n\n','\n','.',','],
        chunk_size = 1000,
        chunk_overlap = 200
    )
    main_placeholder.text("Splitting the documents...")
    docs = text_splitter.split_documents(data)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vectorstore_google = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Building Embedding Vectors...")
    time.sleep(2)
    vectorstore_google.save_local(filename)

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(filename):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        vectorstore = FAISS.load_local(
            filename, embeddings, allow_dangerous_deserialization=True
        )
        retrieval_qa_chat_prompt = ChatPromptTemplate.from_template(
            """ Answer the following questions based only on the provided context.
            <context>
            {context}
            </context>
            Question: {input}"""
        )

        combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
        rag_chain = create_retrieval_chain(vectorstore.as_retriever(), combine_docs_chain)
        result = rag_chain.invoke({"input": query}, return_only_outputs=True)
        st.header("The result is: ")
        st.subheader(result['answer'].strip())

        # sources = result.get("sources",",")
        # if sources:
        #     st.subheader("Sources:")
        #     sources_list = sources.split("\n")
        #     for source in sources_list:
        #         st.write(source)