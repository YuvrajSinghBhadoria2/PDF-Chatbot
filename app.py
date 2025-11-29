import streamlit as st 
from langchain_community.document_loaders import PyPDFLoader
import os 
import pandas as pd
import base64 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
#from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_classic.vectorstores import FAISS
from datetime import datetime
import asyncio
from langchain_core.runnables import RunnableParallel ,RunnablePassthrough
import tempfile


load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


try :
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())



def get_pdf_text(pdf_docs):
    documents = []
    for uploaded_file in pdf_docs:
        # Save uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        
        # Load PDF from temp path
        loader = PyPDFLoader(tmp_file_path)
        docs = loader.load()
        documents.extend(docs)

    return documents


def get_text_chunks(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
    return splitter.split_documents(docs)

def create_vectorstore(splits):
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.from_documents(
        splits,
        embedding=embedding_model 
    )
    db.save_local("faiss_index")
    return db

def conversational_chain():
    llm = ChatGroq(
        model = "llama-3.1-8b-instant",
        groq_api_key = GROQ_API_KEY,
        temperature=0.5
    )

    prompt = PromptTemplate(
        template="""
        Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
        provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """,
        input_variables=['context','question']
    )

    chain = (
        RunnableParallel(
            context=RunnablePassthrough(),
            question=RunnablePassthrough()
        )
        | prompt
        | llm
    )

    return chain

    return chain 
def user_input(user_question,pdf_docs,conversation_history,persist_path = "faiss_index"):
    if not pdf_docs:
        st.warning("Please upload PDF Files before asking teh question ")
        return None 
    
    docs = get_pdf_text(pdf_docs)
    chunks = get_text_chunks(docs)

    db=None
    embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    
    try :
        db = FAISS.load_local(persist_path,embedding,allow_dangerous_deserialization=True)
    except Exception:
        db = None

    if db is None:
        db = FAISS.from_documents(chunks,embedding=embedding)
        db.save_local(persist_path)

    k=4 
    docs_retrieved = db.similarity_search(user_question,4)

    context = "\n\n".join([ doc.page_content for doc in docs_retrieved])

    chain = conversational_chain()

    try:
        result = chain.invoke({"context":context,"question":user_question})
    except Exception as e :
        try:
            result = chain.run({"context": context, "question": user_question})
        except Exception as e2:
            st.error(f"LLM invocation failed: {e} / {e2}")
            return None

    answer_text  =getattr(result,"content",None) or getattr(result, "output", None) or str(result)

    pdf_names = ", ".join({d.metadata.get("source", "") for d in docs_retrieved if d.metadata.get("source")})
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conversation_history.append((user_question, answer_text, "Groq", timestamp, pdf_names))

    return answer_text

def display_chat(conversation_history):
    st.markdown(
        """
        <style>
            .chat-message {padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex;}
            .chat-message.user {background-color: #2b313e;}
            .chat-message.bot {background-color: #475063;}
            .chat-message .avatar {width: 20%;}
            .chat-message .avatar img {max-width: 78px; max-height: 78px; border-radius: 50%; object-fit: cover;}
            .chat-message .message {width: 80%; padding: 0 1.5rem; color: #fff;}
            .chat-message .info {font-size: 0.8rem; margin-top: 0.5rem; color: #ccc;}
        </style>
        """,
        unsafe_allow_html=True
    )

    for question, answer, model_name, timestamp, pdf_name in reversed(conversation_history):
        st.markdown(
            f"""
            <div class="chat-message user">
                <div class="avatar">
                    <img src="https://i.ibb.co/CKpTnWr/user-icon-2048x2048-ihoxz4vq.png">
                </div>
                <div class="message">{question}</div>
            </div>
            <div class="chat-message bot">
                <div class="avatar">
                    <img src="https://i.ibb.co/wNmYHsx/langchain-logo.webp">
                </div>
                <div class="message">{answer}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

# -------------------- Main App --------------------
def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.header("Chat with multiple PDFs (v1) :books:")

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    linkedin_profile_link = "https://www.linkedin.com/in/snsupratim/"
    kaggle_profile_link = "https://www.kaggle.com/snsupratim/"
    github_profile_link = "https://github.com/snsupratim/"

    st.sidebar.markdown(
        f"[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)]({linkedin_profile_link}) "
        f"[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)]({kaggle_profile_link}) "
        f"[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)]({github_profile_link})"
    )

    st.sidebar.title("Menu:")
    col1, col2 = st.sidebar.columns(2)
    reset_button = col2.button("Reset")
    clear_button = col1.button("Rerun")

    if reset_button:
        st.session_state.conversation_history = []

    pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question and pdf_docs:
        user_input(user_question, pdf_docs, st.session_state.conversation_history)

    # Display conversation
    display_chat(st.session_state.conversation_history)

    # Download conversation as CSV
    if st.session_state.conversation_history:
        df = pd.DataFrame(st.session_state.conversation_history, columns=["Question", "Answer", "Model", "Timestamp", "PDF Name"])
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="conversation_history.csv"><button>Download conversation history as CSV</button></a>'
        st.sidebar.markdown(href, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
