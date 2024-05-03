from __future__ import print_function
from pathlib import Path
import os.path
import streamlit as st
import textwrap
import re

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/drive'] 
MAX_TEXT_LEN = 1000
def extract_document_id(url):
    try:
        return url.split('/')[5]
    except IndexError:
        raise ValueError("Invalid Google Docs URL")

def read_paragraph_element(element):
    """Returns the text in the given ParagraphElement.

        Args:
            element: a ParagraphElement from a Google Doc.
    """
    text_run = element.get('textRun')
    if not text_run:
        return ''
    return text_run.get('content')


def read_structural_elements(elements):
    """Recurses through a list of Structural Elements to read a document's text where text may be
        in nested elements.

        Args:
            elements: a list of Structural Elements.
    """
    text = ''
    for value in elements:
        if 'paragraph' in value:
            elements = value.get('paragraph').get('elements')
            for elem in elements:
                text += read_paragraph_element(elem)
        elif 'table' in value:
            # The text in table cells are in nested Structural Elements and tables may be
            # nested.
            table = value.get('table')
            for row in table.get('tableRows'):
                cells = row.get('tableCells')
                for cell in cells:
                    text += read_structural_elements(cell.get('content'))
        elif 'tableOfContents' in value:
            # The text in the TOC is also in a Structural Element.
            toc = value.get('tableOfContents')
            text += read_structural_elements(toc.get('content'))
    return text

def number_of_words(sentence):
    return len(re.findall(r'\w+', sentence))

def textwrap_max_len(text,max_len):
    if (number_of_words(text) > max_len): 
        return textwrap.wrap(text, max_len)
    else:
        return [text]

def data_max_len(data,max_len):
    new_data = [data.pop(0)]
    for d in data:
        if(number_of_words(new_data[-1]) + number_of_words(d) > max_len):
            new_data.append(d)
        else:    
            new_data[-1] = new_data[-1] + "\n\n" + d

    return new_data

def create_gdoc(creds,title):
    service = build('drive', 'v3', credentials=creds)
    doc_metadata = {
        'name': title,
        'parents': ['root'],
        'mimeType': 'application/vnd.google-apps.document'
    }
    doc = service.files().create(body=doc_metadata).execute()
    return doc['id']  

def write_gdoc(creds,doc_id,content):
    service = build('docs', 'v1', credentials=creds)  
    requests = [
         {
            'insertText': {
                'text': content,
                'endOfSegmentLocation': {}
            }
        } 
    ] 

    service.documents().batchUpdate(documentId=doc_id, body={'requests': requests}).execute()
 

def read_gdoc_content(creds,document_id):
    try:
        service = build('docs', 'v1', credentials=creds)

        document = service.documents().get(documentId=document_id).execute()

        doc_content = read_structural_elements(document.get('body').get('content'))
        content = doc_content.strip().split("\n\n")
        data = []
        
        for p in content:
            data = data + textwrap_max_len(p,MAX_TEXT_LEN)

        data = data_max_len(data,MAX_TEXT_LEN)
            
        return data
      
    except HttpError as err:
        print(err)

def read_gdoc_title(creds,document_id):
    service = build('docs', 'v1', credentials=creds)  
    document = service.documents().get(documentId=document_id).execute()
    return document.get('title')

def gdoc_creds():
    """Shows basic usage of the Docs API.
    Prints the title of a sample document.
    """
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    token_path = os.path.dirname(__file__) + '/token.json'
    #credentials_path = os.path.dirname(__file__) + '/credentials.json'
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    else:
        token_data = {"token": st.secrets["GDOCS_TOKEN"], "refresh_token": st.secrets["GDOCS_REFRESH_TOKEN"], "token_uri": "https://oauth2.googleapis.com/token", "client_id": st.secrets["GDOCS_CLIENT_ID"], "client_secret":st.secrets["GDOCS_CLIENT_SECRET"], "scopes": ["https://www.googleapis.com/auth/drive"], "expiry": "2023-04-02T16:06:31.648931Z"}
        creds = Credentials.from_authorized_user_info( token_data, SCOPES)
        
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            #flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
            credentials_config = {"installed":{"client_id":st.secrets["GDOCS_CLIENT_ID"],"project_id":st.secrets["GDOCS_PROJECT_ID"],"auth_uri":"https://accounts.google.com/o/oauth2/auth","token_uri":"https://oauth2.googleapis.com/token","auth_provider_x509_cert_url":"https://www.googleapis.com/oauth2/v1/certs","client_secret":st.secrets["GDOCS_CLIENT_SECRET"],"redirect_uris":["http://localhost"]}}
            
            flow = InstalledAppFlow.from_client_config(credentials_config,SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(token_path, 'w') as token:
            token.write(creds.to_json())
    return creds

def gdoc_set_permission(creds,document_id,email):
    drive_service = build('drive', 'v3', credentials=creds)
    document = drive_service.files().get(fileId=document_id, fields='permissions(id, emailAddress, role)').execute()
    permissions = document.get('permissions', [])

    existing_permission = False
    for p in permissions:
        if 'emailAddress' in p:
            if p['emailAddress'] == email:
                existing_permission = True
                break

    if not existing_permission:
        new_permission = {
            'type': 'user',
            'role': 'writer',
            'emailAddress': email
        }
        drive_service.permissions().create(fileId=document_id, body=new_permission).execute()

     
"""
if __name__ == '__main__':
    creds = gdoc_creds()
    document_id = extract_document_id("https://docs.google.com/document/d/16LE-sjH2y2MM43-OcXmA4JcBfiA-7sIJpCfRxNSC6OI/edit")
    data = read_gdoc_content(creds,document_id)
    title = read_gdoc_title(creds,document_id)

    new_id = create_gdoc(creds,title="OpenAI - " + title)

    print("https://docs.google.com/document/d/"+new_id+"/edit") 
    write_gdoc(creds,new_id,content=data[0])

    gdoc_set_permission(creds,"19BGMFtuhNux2JWU0H_5OiNaQoIIvC2ojPOIZMGR6Ayc","jordan@digitalonda.com")
"""    