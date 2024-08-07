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
 
 
import time
import tiktoken
from split_string import split_string_with_limit
import requests
import json 
import docx2txt
import pdfplumber

ANTHROPIC_API_KEY = st.secrets['ANTHROPIC_API_KEY']   
OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']
PINECONE_API_KEY = st.secrets['PINECONE_API_KEY']
COHERE_API_KEY = st.secrets['COHERE_API_KEY']
  

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

model_name = "gpt-4o"
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
    system_prompt = st.session_state.system_prompt
    
    if not system_prompt:
        system_prompting = "You are a helpful assistant.Based on these documents provided below, please complete the task requested by the user:"
    else:
        system_prompting = system_prompt
        if len(data): 
            system_prompting +='''Improve your answers from previous answers and CONTEXT information below.'''
            system_prompting += "\n [CONTEXT] \n"
            system_prompting += "\n\n".join(data)
    
    our_sms = st.session_state.chat_history["history"]
    our_sms = our_sms[-10:]
    return system_prompting,our_sms

def get_embedding(text,embed_model="text-embedding-3-small" ):
    client = OpenAI(api_key=OPENAI_API_KEY)
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=embed_model).data[0].embedding

default_vec_embedding = get_embedding("default")

def add_to_index(data,nsp="default"):
    data_index.upsert(vectors=data,namespace=nsp)

def get_from_index_raw(vec,top_k=20,nsp="default",filter={}):
    res = data_index.query(vector=vec,top_k=top_k,include_values=True,include_metadata=True,namespace=nsp,
                            filter=filter
                            )
    return res["matches"]
def get_from_index(vec,top_k=20,nsp="default",filter={}):
    res_matches = get_from_index_raw(vec,top_k,nsp,filter)
    docs = [x["metadata"]['text'] for x in res_matches]
    if nsp == "list" or nsp=="chat_history_list":
        docs = { x["metadata"]['doc_id']:x["metadata"]['text'] for i, x in enumerate(res_matches)}
    
    return docs


def get_filter_id(doc_ids):
    return {"doc_id": {"$in": doc_ids}}
 

def get_all_docs():
    docs = get_from_index(default_vec_embedding,1000,"list")
    return docs

def get_all_history_list():
    docs = get_from_index(default_vec_embedding,1000,"chat_history_list")
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

def slugify(s):
  s = s.lower().strip()
  s = re.sub(r'[^\w\s-]', '', s)
  s = re.sub(r'[\s_-]+', '-', s)
  s = re.sub(r'^-+|-+$', '', s)
  return s

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

def load_history(k):
    res_matches = get_from_index_raw(default_vec_embedding,1000,"chat_history",filter={"chat_id":k})
    new_history = {"id": k, "history": []}
    for x in res_matches:
        idx = int(x["metadata"]["order"]) - 1
        new_history["history"].insert(idx,{"role":x["metadata"]["role"],"content":x["metadata"]["text"]} )

    
    st.session_state.chat_history = new_history
     
def delete_docs(doc_id):
    fil =get_filter_id(doc_id)
    l1 = get_from_index_raw(default_vec_embedding,10000,"default",filter=fil) 
    l2 = get_from_index_raw(default_vec_embedding,10000,"list",filter=fil) 
     # delete from index
    d1 = [x["id"] for x in l1]
    d2 = [x["id"] for x in l2]
     
    data_index.delete(d1, namespace="default")
    data_index.delete(d2, namespace="list")
             

 


if not "all_docs" in st.session_state:
    st.session_state.all_docs = {}

all_docs = get_all_docs() 
st.session_state.all_docs = all_docs

def retrive_selected_docs():
    sd = get_from_index_raw(default_vec_embedding,top_k=1,nsp="selected_doc")
    
    if len(sd) > 0:
        sd = sd[0]
        keys = sd["metadata"]["keys"].split(",")
        values = sd["metadata"]["values"].split(",")
        for idx,key in enumerate(keys):
            st.session_state.selected_docs[key] = values[idx]

def save_selected_docs():
    metadata = {"keys": ",".join(st.session_state.selected_docs.keys()),"values": ",".join(st.session_state.selected_docs.values())}
    data = [{ "id": "selected_doc", "values":default_vec_embedding, "metadata": metadata}]
    add_to_index(data, "selected_doc") 

def add_selected_docs(idx,doc_title):
    st.session_state.selected_docs[idx] = doc_title
    save_selected_docs()

if not "selected_docs" in st.session_state:
    st.session_state.selected_docs = {}
retrive_selected_docs()

    
new_doc_modal = Modal(
    "Add New Document", 
    key="new-doc-modal",
    padding=20,    # default value
    max_width=700  # default value
)
if new_doc_modal.is_open():
    with new_doc_modal.container():
        tab0,tab1, tab2 = st.tabs(["Your Documents","Upload Document", "Youtube"])
        with tab0:
            for idx,doc_title in st.session_state.all_docs.items():
                checked = False
                if idx in st.session_state.selected_docs.keys():
                    checked = True
                st.checkbox(doc_title,checked,idx,on_change=add_selected_docs,args=(idx,doc_title) )

           
            if st.button("Delete Selected Documents") :
                if len(st.session_state.selected_docs) > 0:
                    delete_docs(st.session_state.selected_docs.keys())    
                    st.session_state.selected_docs = {}
                    save_selected_docs()
                    new_doc_modal.close()

        with tab1:
            uploaded_file = st.file_uploader("Choose a document file",type=["docx","doc","txt","rtf","pdf"])
            if uploaded_file is not None: 

                if uploaded_file.type == "text/plain":
                    string_data = uploaded_file.read().decode("utf-8")
                elif uploaded_file.type == "application/pdf":
                    pages = pdfplumber.open(uploaded_file).pages
                    l = list(map(lambda page:page.extract_text(),pages))
                    string_data = "\n\n".join(l)
                else:
                    string_data =  docx2txt.process(uploaded_file)    
                
                title = uploaded_file.name
                document_id = slugify(title)
                tiktoken_encoding = tiktoken.get_encoding("cl100k_base")
                chunks = split_string_with_limit(string_data, 512,tiktoken_encoding)
                if document_id in all_docs.keys():
                    st.write("Document already exists.")
                else:
                    with st.spinner(text="Please patient,it may take some time to process the document."):
                        all_docs[document_id] = title
                        st.session_state.selected_docs[document_id] = title
                        st.session_state.all_docs = all_docs 
                        save_doc_to_vecdb(document_id,chunks)
                        save_doc_to_db(document_id,title)
                        st.write("Document added successfully.")
                        new_doc_modal.close()

        with tab2:
            vid_title = st.text_input("Youtube title:")
            vid_url = st.text_input("Enter your Youtube url, Ex: https://www.youtube.com/watch?v=xxxxxx")
            video_id = extract_youtube_id(vid_url)
            submit_video = st.button("Submit Video")

       
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
                               '''You are an elite AI Assistant, renowned for your expertise in providing high-level business consulting and marketing insights to world-class thought leaders and top-tier business professionals across major global cities. Your background includes deep knowledge in strategic planning, market analysis, innovation, and leadership development. You approach each query with a commitment to delivering precise, insightful responses that draw upon your extensive experience and the specific context provided. You continuously refine your answers, ensuring they are tailored to the unique needs of distinguished clients. Always reply directly in the language the question was asked. Ask additional questions if context is needed. If you do not know the answer, simply state that you do not know.'''
                               ) 
  st.session_state.system_prompt = system_prompt
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
         
      bt = st.button(info,key=k)
      if bt:
        load_history(k)
  st.divider()
  delete_history = st.button("Clear History")
  if delete_history:
      data_index.delete(namespace="chat_history", delete_all=True) 
      data_index.delete(namespace="chat_history_list", delete_all=True)   
      st.session_state.all_chat_history = {}       
 
your_prompt = st.chat_input ("Enter your Prompt:" ) 

if your_prompt:
    filter = get_filter_id([doc for doc in st.session_state.selected_docs.keys() ])

    st.session_state.chat_history["history"].append({"role": "user", "content": your_prompt})
    order = len(st.session_state.chat_history["history"])
    
    your_prompt_vec = get_embedding(your_prompt)
    
    if order == 1:
        if st.session_state.chat_history["id"] not in st.session_state.all_chat_history.keys():
            save_his = [{"id":str(st.session_state.chat_history["id"]),"values":your_prompt_vec,"metadata":{ "doc_id":st.session_state.chat_history["id"],"text":your_prompt}}]
            add_to_index(save_his, "chat_history_list")
        st.session_state.all_chat_history[st.session_state.chat_history["id"]] = your_prompt 

    save_prompt = {"id":str(st.session_state.chat_history["id"])+"_"+str(order),"values":your_prompt_vec,"metadata":{"chat_id":st.session_state.chat_history["id"],"order":order,"role":"user","text":your_prompt}}

    data = get_from_index(your_prompt_vec,filter=filter)
    data = cohere_rerank(your_prompt, data)
    
    if api_option == "Anthropic" :
        response = send_llm_claude(data) 
    else:    
        response = send_llm(data)

    st.session_state.chat_history["history"].append({"role": "assistant", "content": response})

    order = len(st.session_state.chat_history["history"])
    save_res = {"id":str(st.session_state.chat_history["id"])+"_"+str(order),"values":get_embedding(response),"metadata":{"chat_id":st.session_state.chat_history["id"],"order":order,"role":"assistant","text":response}}
    add_to_index([save_prompt,save_res], "chat_history")
     
for item in st.session_state.chat_history["history"]:
    if item["role"] == "user" or item["role"] == "assistant":    
        st.chat_message(item["role"]).write(item["content"])
    
    
