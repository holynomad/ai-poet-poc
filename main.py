#2023-08-26 조코딩 온라인특강 : 랭체인 & streamlit 이용한 AI 웹서비스

#from dotenv import load_dotenv
#load_dotenv() # api-key를 숨기는 도구 라이브러리

#from langchain.chat_models import ChatOpenAI
#chat_model = ChatOpenAI()

#result = chat_model.predict("my name is macgyver !!")
#print(result)


#아래는 인공지능 시인 코드 using langchain + openAI api (유료)
# from dotenv import load_dotenv
# load_dotenv()
# import streamlit as st
# from langchain.chat_models import ChatOpenAI

# chat_model = ChatOpenAI()

# st.title('인공지능 시인(GPT-3.5)')

# content = st.text_input('시의 주제를 제시해주세요.')

# if st.button('시 작성 요청하기'):
#     with st.spinner('시 작성 중...'):
#         result = chat_model.predict(content + "에 대한 시를 써줘")
#         st.write(result)



#아래는 ggml + ctransformers 무료 인공지능 LLM 로컬환경 구축 (llama는 한글아직 약함..)
# import streamlit as st
# from langchain.chat_models import ChatOpenAI
# from langchain.llms import CTransformers

# llm = CTransformers(
#   model="llama-2-7b-chat.ggmlv3.q2_K.bin",
#   model_type="llama"
# )

# st.title('인공지능 시인(LLaMa)')

# content = st.text_input('시의 주제를 제시해주세요.')

# if st.button('시 작성 요청하기'):
#   with st.spinner('시 작성중 ...'):
#     result = llm.predict("write a poen about " + content + " : ")
#     st.write(result)


#아래는 chatPDF using openAI API (유료) 모델
# from dotenv import load_dotenv
# load_dotenv()
# from langchain.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import Chroma
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.chat_models import ChatOpenAI
# from langchain.retrievers.multi_query import MultiQueryRetriever
# from langchain.chains import RetrievalQA
# import streamlit as st
# import tempfile
# import os

# #제목
# st.title("ChatPDF")
# st.write("---")

# #파일 업로드
# uploaded_file = st.file_uploader("Choose a file")
# st.write("---")

# def pdf_to_document(uploaded_file):
#     temp_dir = tempfile.TemporaryDirectory()
#     temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
#     with open(temp_filepath, "wb") as f:
#         f.write(uploaded_file.getvalue())
#     loader = PyPDFLoader(temp_filepath)
#     pages = loader.load_and_split()
#     return pages

# #업로드 되면 동작하는 코드
# if uploaded_file is not None:
#     pages = pdf_to_document(uploaded_file)

#     #Split
#     text_splitter = RecursiveCharacterTextSplitter(
#         # Set a really small chunk size, just to show.
#         chunk_size = 300,
#         chunk_overlap  = 20,
#         length_function = len,
#         is_separator_regex = False,
#     )
#     texts = text_splitter.split_documents(pages)

#     #Embedding
#     embeddings_model = OpenAIEmbeddings()

#     # load it into Chroma
#     db = Chroma.from_documents(texts, embeddings_model)

#     #Question
#     st.header("PDF에게 질문해보세요!!")
#     question = st.text_input('질문을 입력하세요')

#     if st.button('질문하기'):
#         with st.spinner('Wait for it...'):
#             llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
#             qa_chain = RetrievalQA.from_chain_type(llm,retriever=db.as_retriever())
#             result = qa_chain({"query": question})
#             st.write(result["result"])


#아래는 chatPDF 최종 수익화 모델(?).. 정확히 BM이 무엇인지는 모르겠음
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import streamlit as st
import tempfile
import os
from streamlit_extras.buy_me_a_coffee import button

button(username="jocoding", floating=True, width=221)

#제목
st.title("ChatPDF")
st.write("---")

#OpenAI KEY 입력 받기
openai_key = st.text_input('OPEN_AI_API_KEY', type="password")

#파일 업로드
uploaded_file = st.file_uploader("PDF 파일을 올려주세요!",type=['pdf'])
st.write("---")

def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages

#업로드 되면 동작하는 코드
if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)

    #Split
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 300,
        chunk_overlap  = 20,
        length_function = len,
        is_separator_regex = False,
    )
    texts = text_splitter.split_documents(pages)

    #Embedding
    embeddings_model = OpenAIEmbeddings(openai_api_key=openai_key)

    # load it into Chroma
    db = Chroma.from_documents(texts, embeddings_model)

    #Question
    st.header("PDF에게 질문해보세요!!")
    question = st.text_input('질문을 입력하세요')

    if st.button('질문하기'):
        with st.spinner('Wait for it...'):
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_key)
            qa_chain = RetrievalQA.from_chain_type(llm,retriever=db.as_retriever())
            result = qa_chain({"query": question})
            st.write(result["result"])
