import sys
import os, docx2txt
from htmlTemplates import css, bot_template, user_template
import streamlit as st
from langchain.memory import VectorStoreRetrieverMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from PyPDF2 import PdfReader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.prompts import PromptTemplate
def handle_userinput(user_question):
    response = st.session_state.conversation({'query': user_question})
    print ("Printing response at handle userinput")
    print(response)
    
    st.write(bot_template.replace("{{MSG}}", response['result']), unsafe_allow_html=True)
def get_conversation_chain(vectorstore,QA_CHAIN_PROMPT):
    model_path = "/run/media/abhilash/data2/models/orca-mini-13b.ggmlv3.q8_0.bin"
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    callbacks = [StreamingStdOutCallbackHandler()]
    llm = GPT4All(model=model_path, max_tokens= 32000, backend="llama", 
                  callbacks=callbacks,
                  verbose=False)
    #memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":4})
    conversation_chain= RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", 
                                                    retriever=retriever,
                                                    #retriever=vectorstore.as_retriever(),
                                                    return_source_documents=False,
                                                    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
    return conversation_chain
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_texts(text_chunks,embeddings,persist_directory='db')
    vectorstore.persist()
    return vectorstore
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        )
    chunks = text_splitter.split_text(text)
    return chunks
def get_doc_text(docs):
    text = ""
    for doc in docs:
        if doc.name.endswith(".pdf"):
            pdf_reader = PdfReader(doc)
            for page in pdf_reader.pages:
                text += page.extract_text()
        if doc.name.endswith(".docx"):
            text += docx2txt.process(doc)
        if doc.name.endswith(".txt"):
            raw_text = str(doc.read(),"utf-8")
            text +=raw_text
    return text
def main():
    #load_dotenv()
    template = """Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}
    Question: {question}

    Only return the helpful answer below and nothing else.
    Helpful answer:
    """
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template,)
    st.set_page_config(page_title="Chat with multiple Documents", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    st.header("Chat on ProcessPAD")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)
    with st.sidebar:
        st.subheader("Your documents")
        docs = st.file_uploader("Upload your Documents here and click on 'Process'", 
                                accept_multiple_files=True, type=['txt','docx','pdf'])
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                all_text = get_doc_text(docs)
                # get the text chunks
                text_chunks = get_text_chunks(all_text)
                # create vector store
                vectorstore = get_vectorstore(text_chunks)
                # create conversation chain
                print('Done Processing')
                st.session_state.conversation = get_conversation_chain(vectorstore,QA_CHAIN_PROMPT)
if __name__ == '__main__':
    main()