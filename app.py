 
from openai import OpenAI
import streamlit as st
from gdocs import gdocs
from streamlit_modal import Modal
from pinecone import Pinecone
import cohere
  
# init client
COHERE_API_KEY = st.secrets('COHERE_API_KEY')
cohere_client = cohere.Client(COHERE_API_KEY)
def cohere_rerank(query: str,docs, top_n=3):
    rerank_docs = cohere_client.rerank(
    query=query, documents=docs, top_n=top_n,return_documents=True, model="rerank-english-v2.0"
    ) 
    return [doc.document.text for doc in rerank_docs.results]

pc = Pinecone(st.secrets('PINECONE_API_KEY'))
data_index = pc.Index("chatdoc")

OPENAI_API_KEY = st.secrets('OPENAI_API_KEY')
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

embed_model = "text-embedding-3-large"

def get_embedding(text ):
   client = OpenAI(
        api_key=OPENAI_API_KEY,
    )
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=embed_model).data[0].embedding

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

with st.sidebar:
  add_new_doc = st.button("Add New Document")
  if add_new_doc:
    new_doc_modal.open()

  st.subheader("Select Your Documents")
     
  doc_options = st.multiselect(
    'Select the documents to query',
    all_docs.keys(),
    format_func = lambda x: all_docs[x] if x in all_docs else x,
    )
if not "the_last_reply" in st.session_state:
    st.session_state.the_last_reply = ""
if not "the_last_prompt" in st.session_state:
    st.session_state.the_last_prompt = ""
        
#your_prompt = st.text_area("Enter your Prompt:" ) 
your_prompt = st.chat_input ("Enter your Prompt:" ) 
#submit_llm = st.button("Send")
if your_prompt:
    filter = get_filter_id([doc for doc in doc_options])
    data = get_from_index(your_prompt,filter=filter)
    if len(data) > 0 :
        data = cohere_rerank(your_prompt, data)
        st.session_state.the_last_prompt = your_prompt
        response = send_llm(your_prompt,data)
        st.session_state.the_last_reply = response.content
        st.write(response.content)
    else:
        st.write("Sorry, no documents selected.")