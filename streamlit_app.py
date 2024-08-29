import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from PIL import Image
import datetime
from langgraph.graph import END, StateGraph
from langchain_groq import ChatGroq
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.prompts import PromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from typing_extensions import TypedDict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
import pickle, os
from langchain_community.embeddings import GPT4AllEmbeddings
# from dotenv import load_dotenv
import parameter
# load_dotenv()
wrapper = DuckDuckGoSearchAPIWrapper(max_results=5)
web_search_tool = DuckDuckGoSearchRun(api_wrapper=wrapper)

try:
    with open("data_chat.pkl", 'rb') as fp:
        chat_history = pickle.load(fp)
        # print(chat_history)
except:
    chat_history = []
try:
    with open("template.pkl", 'rb') as fp:
        template = pickle.load(fp)
except:
    pass
# print(template["ROUTE_TEMPLATE"])
router_prompt = PromptTemplate(
    template=template["ROUTE_TEMPLATE"],
    input_variables=["question"],
)

generate_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", template["GENERATE_TEMPLATE"] ,
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)
# print(template["QUERY_TEMPLATE"])
query_prompt = PromptTemplate(
    template=template["QUERY_TEMPLATE"],
    input_variables=["question"],
)

remind_prompt = PromptTemplate(
    template=template["SCHEDULE_TEMPLATE"],
    input_variables=["time"],
)
class State(TypedDict):

    question : str
    generation : str
    search_query : str
    context : str


dt = datetime.datetime.now()
formatted = dt.strftime("%B %d, %Y %I:%M:%S %p")
image_bot = Image.open("avata/avata_bot.png")
image_human = Image.open("avata/avata_human.png")

# local_llm = 'aleni_ox'

# llm = ChatOllama(model=local_llm,
#                  keep_alive="3h", 
#                  max_tokens=512,  
#                  temperature=0,
#                 # callbacks=[StreamingStdOutCallbackHandler()]
#                 )

llm = ChatGroq(temperature=0, model_name="llama-3.1-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
embedding_model = GPT4AllEmbeddings(model_name="all-MiniLM-L6-v2.gguf2.f16.gguf", gpt4all_kwargs={'allow_download': 'True'})

question_router = router_prompt | llm | JsonOutputParser()
generate_chain = generate_prompt | llm | StrOutputParser()
query_chain = query_prompt | llm | JsonOutputParser()
remind_chain = remind_prompt | llm | StrOutputParser()

def Agent():
    workflow = StateGraph(State)
    workflow.add_node("websearch", web_search)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("generate", generate)

    # Build the edges
    workflow.set_conditional_entry_point(
        route_question,
        {
            "websearch": "transform_query",
            "generate": "generate",
        },
    )
    workflow.add_edge("transform_query", "websearch")
    workflow.add_edge("websearch", "generate")
    workflow.add_edge("generate", END)

    # Compile the workflow
    local_agent = workflow.compile()
    return local_agent

def transform_query(state):
    print("Step: Tối ưu câu hỏi của người dùng")
    question = state['question']
    gen_query = query_chain.invoke({"question": question})
    print(gen_query)
    search_query = gen_query["query"]
    return {"search_query": search_query}

def web_search(state):
    search_query = state['search_query']
    print(f'Step: Đang tìm kiếm web cho: "{search_query}"')
    
    # Web search tool call
    search_result = web_search_tool.invoke(search_query)
    print("Search result:", search_result)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunked_documents = text_splitter.split_text(search_result)
    vectorstore = FAISS.from_texts(chunked_documents, embedding_model)
    search_result = vectorstore.similarity_search(search_query)
    return {"context": search_result}

def route_question(state):
    print("Step: Routing Query")
    question = state['question']
    try:
        output = question_router.invoke({"question": question})
    except:
        return "generate"
    print('Lựa chọn của AI là: ', output)
    if output['choice'] == "web_search":
        # print("Step: Routing Query to Web Search")
        return "websearch"
    elif output['choice'] == 'generate':
        # print("Step: Routing Query to Generation")
        return "generate"
def generate(state):    
    print("Step: Đang tạo câu trả lời từ những gì tìm được")
    question = state["question"]
    context = state["context"]
    return {'question': question, 'context': context}

def plan_in_day():
    for chunk in remind_chain.stream({"time": formatted}):
        print(chunk, end="", flush=True)
        # st.session_state["full_message"] += chunk
        yield chunk

# def convert_str(list_content):
#     list_content 

def search_tool(prompt:str):
    if "/bird:" in prompt: 
        context = parameter.vectorstore_text.similarity_search(prompt)
        questions = prompt.replace("/bird:","")
        print(context, prompt.replace("/bird:",""))
    elif "/owl:" in prompt:
        context = parameter.vectorstore_pdf.similarity_search(prompt)
        print(context)
        questions = prompt.replace("/owl:","")
    elif "/monkey:" in prompt:
        context = parameter.vectorstore_STT.similarity_search(prompt)
        print(context, prompt.replace("/monkey:", ""))
        questions = prompt.replace("/monkey:","")

    elif "/worlf:" in prompt:
        context = parameter.vectorstore_csv.similarity_search(prompt)
        print(context, prompt.replace("/worlf:", ""))
        questions = prompt.replace("/worlf:","")
    else:
        local_agent = Agent()
        output = local_agent.invoke({"question": prompt})
        context = output['context']
        questions = output['question']        
    return {"context": context, "question": questions}
    # else:

def generate_response(prompt):
    # local_agent = Agent()
    # output = local_agent.invoke({"question": prompt})
    output = search_tool(prompt)
    context = output['context']
    questions = output['question']
    # print(parameter.vectorstore_csv, parameter.vectorstore_text, parameter.vectorstore_pdf, parameter.vectorstore_STT)
    for chunk in generate_chain.stream({"context": context, "question": questions, "chat_history": chat_history[-100:]}):
        print(chunk, end="", flush=True)
        st.session_state["full_message"] += chunk
        yield chunk
    # print(st.session_state["full_message"])
    chat_history.append(HumanMessage(content=questions))
    chat_history.append(AIMessage(content=st.session_state["full_message"]))
    with open('data_chat.pkl', 'wb') as fp:
        pickle.dump(chat_history, fp)

def create_vectorstore(data):
    from langchain_community.document_loaders import WebBaseLoader
    import urllib.parse

    parsed_url = urllib.parse.urlparse(data)
    if parsed_url.scheme and parsed_url.netloc:
        loader = WebBaseLoader(data)
        data = loader.load()
        # print(data)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        chunked_documents = text_splitter.split_documents(data)
        vectorstore = FAISS.from_documents(chunked_documents, embedding_model)
    else:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        chunked_documents = text_splitter.split_text(data)
        vectorstore = FAISS.from_texts(chunked_documents, embedding_model)

    return vectorstore


def main():

    st.set_page_config(page_title="Abox", page_icon=":speech_balloon:")
    st.title("💬 Chatbot")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    #     with st.spinner("Đang lên kế hoạch..."):        
    #         st.write_stream(plan_in_day)
        # {"role": "assistant", "content": "Anh cần tôi giúp gì nào"}]

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.chat_message(msg["role"], avatar=image_human).write(msg["content"])
        else:
            st.chat_message(msg["role"], avatar=image_bot).write(msg["content"])
    with st.sidebar:
        tab1, tab2, tab3, tab4 = st.tabs(["Bird📄", "Owl📚", "Worlf🔍", "Monkey🗣️"])
        # st.balloons()

        with tab1:
            text = st.text_area("Nhập đoạn văn bản:")
            # if len(text) > 0:
            if text:
                with st.spinner(text="In progress..."):
                    # global vectorstore_text 
                    parameter.vectorstore_text = create_vectorstore(text)

        with tab2:
            # from io import StringIO
            from PyPDF2 import PdfReader
            content_pdf = ""
            uploaded_file = st.file_uploader("Choose documents", accept_multiple_files=True, type=["PDF"])
            uploaded_file
            if uploaded_file:
                with st.spinner('Wait for it...'):
                    for pdf in uploaded_file:
                        print(pdf.name)
                        pdf_loader = PdfReader(pdf)
                        for page in pdf_loader.pages:
                            content_pdf += page.extract_text()
                    print(content_pdf)
                    # global vectorstore_pdf 
                    parameter.vectorstore_pdf = create_vectorstore(content_pdf)

        with tab3:
            import pandas as pd
            from langchain_core.documents import Document
            # import seaborn as sns
            uploaded_file = st.file_uploader("Choose a file", accept_multiple_files=False, type=["csv", "png", "jpg", "jpeg"])
            if uploaded_file is not None:

                # daf = pd.read_csv(uploaded_file)
                # st.data_editor(uploaded_file)
                with st.spinner('Wait for it...'):
                    # for file_csv in uploaded_csv:
                    if ".csv" in uploaded_file.name:
                        document = []
                        df = pd.read_csv(uploaded_file)
                        st.data_editor(df)
                        for index, data in df.iterrows():
                            metadata={'source': uploaded_file.name, 'row': index}
                            page_content = ''
                            for feauture in data.index:
                                page_content += "\n" + str(feauture) + ": " + str(data[feauture])
                            document.append(Document(metadata=metadata, page_content=page_content)) 
                        parameter.vectorstore_csv = FAISS.from_documents(document, embedding_model)
                    # else:
                    #     st.image(uploaded_file, caption="Image uploaded")
                    #     image = Image.open(uploaded_file).resize((256,256))

            with tab4:
                uploaded_media = st.file_uploader("Choose a video or audio", accept_multiple_files=False, type=["mp4", "mp3", "wav", "..."])
                # uploaded_media
                # if st.button("speech to text"):
                if uploaded_media is not None:
                    if ".mp4" not in uploaded_media.name:
                        st.audio(uploaded_media)
                        from groq import Groq
                        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
                        # filename = "9. Writing T1 - Maps - Homework - Track 1.mp3"
                        # with open(filename, "rb") as file:
                        with st.spinner('Wait for it...'):
                            transcription = client.audio.transcriptions.create(
                            file=uploaded_media,
                            model="whisper-large-v3",
                            response_format="verbose_json",
                            )
                            transcription.text
                            # global vectorstore_STT 
                            parameter.vectorstore_STT = create_vectorstore(transcription.text)
                    else:
                        st.video(uploaded_media)
    # prompt = 
    if prompt:=st.chat_input("Nhập câu hỏi bạn muốn hỏi vào đây!"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user", avatar=image_human).write(prompt)
        st.session_state["full_message"] = ""
        with st.spinner("Đang tạo câu trả lời..."):  
            st.chat_message("assistant", avatar=image_bot).write_stream(generate_response(prompt))
        st.session_state.messages.append({"role": "assistant", "content": st.session_state["full_message"]})   




if __name__ == "__main__":
    main()
