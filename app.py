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
#from pinecone_text.sparse import BM25Encoder
#from pinecone_text.dense import OpenAIEncoder
#from pinecone_text.hybrid import hybrid_convex_scale
from pymongo.mongo_client import MongoClient
 
import tiktoken
from split_string import split_string_with_limit

ANTHROPIC_API_KEY = st.secrets['ANTHROPIC_API_KEY']   
OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']
PINECONE_API_KEY = st.secrets['PINECONE_API_KEY']
COHERE_API_KEY = st.secrets['COHERE_API_KEY']
MONGODB_API_KEY = st.secrets['MONGODB_API_KEY']
MONGODB_API_APPNAME = st.secrets['MONGODB_API_APPNAME']
#uri = f"mongodb+srv://{MONGODB_API_KEY}@cluster0.t7fr2hb.mongodb.net/?retryWrites=true&w=majority&appName={MONGODB_API_APPNAME}"

#BM25Encoder = BM25Encoder()
#BM25Encoder.load("./msmarco_bm25_params.json")

#OpenAIEncoder = OpenAIEncoder() 

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
def send_llm(prompt,data):
    last_prompt = st.session_state.the_last_prompt
    last_reply = st.session_state.the_last_reply
    
    system_prompting = "You are a helpful assistant."
    if len(data):
        system_prompting += "Based on these documents provided below, please complete the task requested by the user:" 
        system_prompting += "\n\n".join(data)
            
    client = OpenAI(
        api_key=OPENAI_API_KEY,
    )
    our_sms = []
    our_sms.append({"role": "system", "content": system_prompting })
    if last_prompt != "":
        our_sms.append( {"role": "user", "content": last_prompt})
    if last_reply != "":
        our_sms.append( {"role": "assistant", "content": last_reply})
    our_sms.append( {"role": "user", "content": prompt})
    chat_completion = client.chat.completions.create(
        messages=our_sms,
        model=model_name,
    )
    return chat_completion.choices[0].message

def send_llm_claude(prompt,data):
    last_prompt = st.session_state.the_last_prompt
    last_reply = st.session_state.the_last_reply
    
    system_prompting = "You are a helpful assistant."
    if len(data):
        system_prompting += "Based on these documents provided below, please complete the task requested by the user:" 
        system_prompting += "\n\n".join(data)
    our_sms = []
     
    our_sms.append( {"role": "user", "content": prompt})

    message = client_claude.messages.create(
    model="claude-3-5-sonnet-20240620",
    max_tokens=4096,
    system = system_prompting,
    messages=our_sms
    )
    return message.content[0].text


def add_to_index(data,nsp="default"):
    data_index.upsert(vectors=data,namespace=nsp)

def get_from_index(prompt,top_k=20,nsp="default",filter={}):
    data = get_embedding(prompt)
    res = data_index.query(vector=data,top_k=top_k,include_values=True,include_metadata=True,namespace=nsp,
                            filter=filter
                            )
    docs = [x["metadata"]['text'] for x in res["matches"]]
    if nsp == "list":
        docs = { x["metadata"]['doc_id']:x["metadata"]['text'] for i, x in enumerate(res["matches"])}
    return docs

def get_filter_id(doc_ids):
    return {"doc_id": {"$in": doc_ids}}

def get_all_docs():
    docs = get_from_index("document",1000,"list")
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
   client = OpenAI(
        api_key=OPENAI_API_KEY,
    )
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=embed_model).data[0].embedding
'''
def get_query_embedding(text):
    dense_vector = OpenAIEncoder.encode_queries([text])
    sparse_vector = BM25Encoder.encode_queries(text)
    hybrid_dense, _ = hybrid_convex_scale(dense_vector[0], sparse_vector, alpha=0.8)
    return hybrid_dense

def get_document_embedding(text):
    dense_vector = OpenAIEncoder.encode_documents([text])
    sparse_vector = BM25Encoder.encode_documents(text)
    hybrid_dense, _ = hybrid_convex_scale(dense_vector[0], sparse_vector, alpha=0.8)
    return hybrid_dense
'''
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
        tab1, tab2 = st.tabs(["Gooogle Docs", "Youtube"])
        with tab1:
            doc_url = st.text_input("Enter your Gooogle Docs url:")
            submit_button = st.button("Submit")
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
                        new_doc_modal.close()
        with tab2:
            vid_title = st.text_input("Youtube title:")
            vid_url = st.text_input("Enter your Youtube url, Ex: https://www.youtube.com/watch?v=fflkFtIwQXo")
            video_id = extract_youtube_id(vid_url)
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



with st.sidebar:
  add_new_doc = st.button("Add New Document")
  if add_new_doc:
    new_doc_modal.open()

  '''st.subheader("Select Your Documents")
     
  doc_options = st.multiselect(
    'Select the documents to query',
    all_docs.keys(),
    format_func = lambda x: all_docs[x] if x in all_docs else x,
    )'''
  api_option = st.selectbox(
    'Select the API',
    ('OpenAI', 'Anthropic'),
    )
  st.subheader("Your Documents")
  for doc_title in all_docs.keys():
      st.text(doc_title)
      st.divider()
  
if not "the_last_reply" in st.session_state:
    st.session_state.the_last_reply = ""
if not "the_last_prompt" in st.session_state:
    st.session_state.the_last_prompt = ""
        
#your_prompt = st.text_area("Enter your Prompt:" ) 
your_prompt = st.chat_input ("Enter your Prompt:" ) 
#submit_llm = st.button("Send")
if your_prompt:
    #filter = get_filter_id([doc for doc in doc_options])
    data = get_from_index(your_prompt)
    if len(data) > 0 :
        data = cohere_rerank(your_prompt, data)
        st.session_state.the_last_prompt = your_prompt
        if api_option == "Anthropic" :
            response = send_llm_claude(your_prompt,data)
            st.session_state.the_last_reply = response
            with st.chat_message("user"):
                st.write(your_prompt)
            with st.chat_message("assistant"):
                st.write(response)
        else:    
            response = send_llm(your_prompt,data)
            st.session_state.the_last_reply = response.content
            
            with st.chat_message("user"):
                st.write(your_prompt)
            with st.chat_message("assistant"):
                st.write(response.content)
    else:
        st.write("Sorry, no documents selected.")
