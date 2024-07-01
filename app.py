from openai import OpenAI
import streamlit as st
from gdocs import gdocs
from streamlit_modal import Modal
from pinecone import Pinecone
import cohere
from anthropic import Anthropic  
import re
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
 
from pymongo.mongo_client import MongoClient
import time
import tiktoken
from split_string import split_string_with_limit
import requests
import json 

ANTHROPIC_API_KEY = st.secrets['ANTHROPIC_API_KEY']   
OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']
PINECONE_API_KEY = st.secrets['PINECONE_API_KEY']
COHERE_API_KEY = st.secrets['COHERE_API_KEY']
MONGODB_API_KEY = st.secrets['MONGODB_API_KEY']
MONGODB_API_APPNAME = st.secrets['MONGODB_API_APPNAME']

DB_SERVICE_KEY = st.secrets['DB_SERVICE_KEY']
DB_SERVICE_URL = st.secrets['DB_SERVICE_URL']

 


def mongodb_client():
    uri = f"mongodb+srv://{MONGODB_API_KEY}@cluster0.t7fr2hb.mongodb.net/?retryWrites=true&ssl=true&w=majority&appName={MONGODB_API_APPNAME}"
    client = MongoClient(uri)
    return client

def save_history_to_db(data):
    pass

cohere_client = cohere.Client(COHERE_API_KEY)
def cohere_rerank(query: str,docs, top_n=3):
    rerank_docs = cohere_client.rerank(
    query=query, documents=docs, top_n=top_n,return_documents=True, model="rerank-english-v2.0"
    ) 
    return [doc.document.text for doc in rerank_docs.results]

client_claude = Anthropic(
    api_key=ANTHROPIC_API_KEY
)

pc = Pinecone(PINECONE_API_KEY)
data_index = pc.Index("chatdoc")

model_name = "gpt-4-turbo-preview"
def send_llm(data):
    system_prompting,messages = get_llm_prompt(data)
    client = OpenAI(
        api_key=OPENAI_API_KEY,
    )
    our_sms = [{"role": "system", "content": system_prompting }]
    our_sms.extend(messages)
     
    chat_completion = client.chat.completions.create(
        messages=our_sms,
        model=model_name,
    )
    return chat_completion.choices[0].message.content

def send_llm_claude(data):
    system_prompting,our_sms = get_llm_prompt(data)
    message = client_claude.messages.create(
    model="claude-3-5-sonnet-20240620",
    max_tokens=4096,
    system = system_prompting,
    messages=our_sms
    )
    return message.content[0].text

def get_llm_prompt(data):
    if not system_prompt:
        system_prompting = "You are a helpful assistant.Based on these documents provided below, please complete the task requested by the user:"
    else:
        system_prompting = system_prompt
        system_prompting += "\n [CONTEXT] \n"
        system_prompting += "\n\n".join(data)
    
    our_sms = st.session_state.chat_history["history"]
    our_sms = our_sms[-10:]
    return system_prompting,our_sms

def add_to_index(data,nsp="default"):
    data_index.upsert(vectors=data,namespace=nsp)

def get_from_index(vec,top_k=20,nsp="default",filter={}):
    res = data_index.query(vector=vec,top_k=top_k,include_values=True,include_metadata=True,namespace=nsp,
                            filter=filter
                            )
    docs = [x["metadata"]['text'] for x in res["matches"]]
    if nsp == "list" or nsp=="chat_history_list":
        docs = { x["metadata"]['doc_id']:x["metadata"]['text'] for i, x in enumerate(res["matches"])}
    return docs

def get_filter_id(doc_ids):
    return {"doc_id": {"$in": doc_ids}}
 

def get_all_docs():
    docs = get_from_index(get_embedding("document"),1000,"list")
    return docs

def get_all_history_list():
    docs = get_from_index(get_embedding("history"),1000,"chat_history_list")
    return docs
    
def save_doc_to_db(document_id,title):
    metadata = {"doc_id": document_id,"text": title}
    data = [{ "id": document_id, "values":get_embedding(title), "metadata": metadata}]
    add_to_index(data, "list")

def save_doc_to_vecdb(document_id,chunks):
    data = []
    lim = 100
    for idx,chunk in enumerate(chunks):
        metadata = {"doc_id": document_id,"text": chunk}
        data.append({ "id": document_id+"_"+str(idx),"values":get_embedding(chunk),"metadata": metadata})
        if len(data) >= lim:
            add_to_index(data)
            data = []
                
    if len(data) > 0 :
        add_to_index(data)

def get_gdoc(url):
    creds = gdocs.gdoc_creds()
    document_id = gdocs.extract_document_id(url)
    chunks = gdocs.read_gdoc_content(creds,document_id)
    title = gdocs.read_gdoc_title(creds,document_id)
    return document_id,title,chunks

def extract_youtube_id(url):
    pattern = (
        r'(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/|youtube\.com/v/|youtube\.com/shorts/|youtube\.com/clip/|youtube\.com/user/.*#p/u/\d+/|youtube\.com/attribution_link\?a=|youtube\.com/.*#.*|youtube\.com/live/|youtube\.com/video/|youtube\.com/clip/|youtube\.com/.*#.*|youtube\.com/user/[^/]+/.*#.*|youtube\.com/[^/]+/[^/]+/[^/]+/|youtube\.com/.*\?v=|youtube\.com/.*\?clip_id=)([a-zA-Z0-9_-]{11})'
    )
    # Search for the pattern in the URL
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    else:
        return None
 
def get_embedding(text,embed_model="text-embedding-3-small" ):
    client = OpenAI(api_key=OPENAI_API_KEY)
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=embed_model).data[0].embedding
 
if not "all_docs" in st.session_state:
    st.session_state.all_docs = {}
all_docs = get_all_docs()
 
st.session_state.all_docs = all_docs
 
new_doc_modal = Modal(
    "Add New Document", 
    key="new-doc-modal",
    padding=20,    # default value
    max_width=700  # default value
)
if new_doc_modal.is_open():
    with new_doc_modal.container():
        tab0,tab1, tab2 = st.tabs(["Your Documents","Gooogle Docs", "Youtube"])
        with tab0:
            for doc_title in all_docs.values():
                st.text(doc_title)
        with tab1:
            doc_url = st.text_input("Enter your Gooogle Docs url:")
            submit_button = st.button("Submit")
            
        with tab2:
            vid_title = st.text_input("Youtube title:")
            vid_url = st.text_input("Enter your Youtube url, Ex: https://www.youtube.com/watch?v=xxxxxx")
            video_id = extract_youtube_id(vid_url)
            submit_video = st.button("Submit Video")

        if submit_button:
            with st.spinner(text="Please patient,it may take some time to process the document."):
                document_id,title,chunks = get_gdoc(doc_url)
                if document_id in all_docs.keys():
                    st.write("Document already exists.")
                else:
                    all_docs[document_id] = title
                    save_doc_to_vecdb(document_id,chunks)
                    save_doc_to_db(document_id,title)
                    st.session_state.all_docs = all_docs 
                    doc_url = ""
                    st.write("Document added successfully.")

        if submit_video:
            with st.spinner(text="Please patient,it may take some time to process the document."):
                if not video_id or video_id in all_docs.keys():
                    st.write("Video already exists.")
                else:            
                    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])  
                    formatter = TextFormatter()
                    formatted_transcript = formatter.format_transcript(transcript)
                        
                    save_doc_to_db(video_id,vid_title)
                    all_docs[video_id] = vid_title
                        
                    tiktoken_encoding = tiktoken.get_encoding("cl100k_base")
                    chunks = split_string_with_limit(formatted_transcript, 512,tiktoken_encoding)
                    save_doc_to_vecdb(video_id,chunks)
                    vid_title = ""
                    vid_url = ""
                    st.write("Document added successfully.")
                    st.session_state.all_docs = all_docs        

if not "chat_history" in st.session_state:
    st.session_state.chat_history = {"id":int(time.time()),"history":[]}
if not "all_chat_history" in st.session_state:
    st.session_state.all_chat_history = get_all_history_list()

with st.sidebar:
  #st.subheader("Select Your Documents")  
  #doc_options = st.multiselect('Select the documents to query',all_docs.keys(),format_func = lambda x: all_docs[x] if x in all_docs else x,)
   
  system_prompt = st.text_area("System Prompt",
                               '''You are an AI Assistant that help everyone by answering questions, and improve your answers from previous answers and CONTEXT information below.
Answer in the same language the question was asked.Just reply directly, don't say base on history or mention about chat history.  
If you don't know the answer, just say that you don't know.'''
                               ) 
  api_option = st.selectbox(
    'Select the API',
    ('OpenAI', 'Anthropic'),
    )
 
  add_new_doc = st.button("Your Document")
  if add_new_doc:
    new_doc_modal.open()
 
  st.subheader("Recent")
  allhistories = st.session_state.all_chat_history
  for k in allhistories.keys():
      item = allhistories[k]
      info = (item[:30] + '..') if len(item) > 75 else item
      st.markdown(f"<a id='history_{k}'>{info}</a>", unsafe_allow_html=True)   
      if st.button(key=k):
          st.write(f"{k} clicked")    
  
 
your_prompt = st.chat_input ("Enter your Prompt:" ) 

if your_prompt:
    #filter = get_filter_id([doc for doc in doc_options])

    st.session_state.chat_history["history"].append({"role": "user", "content": your_prompt})
    order = len(st.session_state.chat_history["history"])
    

    your_prompt_vec = get_embedding(your_prompt)
    
    if order == 1:
        st.session_state.all_chat_history[st.session_state.chat_history["id"]] = your_prompt   
        save_his = [{"id":str(st.session_state.chat_history["id"]),"values":your_prompt_vec,"metadata":{ "doc_id":st.session_state.chat_history["id"],"text":your_prompt}}]
        add_to_index(save_his, "chat_history_list")

    save_prompt = {"id":str(st.session_state.chat_history["id"])+"_"+str(order),"values":your_prompt_vec,"metadata":{"chat_id":st.session_state.chat_history["id"],"order":order,"type":"history","text":your_prompt}}

    data = get_from_index(your_prompt_vec)
    data = cohere_rerank(your_prompt, data)
    
    if api_option == "Anthropic" :
        response = send_llm_claude(data) 
    else:    
        response = send_llm(data)

    st.session_state.chat_history["history"].append({"role": "assistant", "content": response})

    order = len(st.session_state.chat_history["history"])
    save_res = {"id":str(st.session_state.chat_history["id"])+"_"+str(order),"values":get_embedding(response),"metadata":{"chat_id":st.session_state.chat_history["id"],"order":order,"type":"history","text":response}}
    add_to_index([save_prompt,save_res], "chat_history")
     
          
for item in st.session_state.chat_history["history"]:
    if item["role"] == "user" or item["role"] == "assistant":    
        st.chat_message(item["role"]).write(item["content"])
    
    
