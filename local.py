import os
from dotenv import load_dotenv
load_dotenv() 
from pinecone_text.sparse import BM25Encoder
from pinecone_text.dense import OpenAIEncoder
from pinecone_text.hybrid import hybrid_convex_scale
import re
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
import tiktoken
from pymongo.mongo_client import MongoClient
from split_string import split_string_with_limit

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
MONGODB_API_KEY = os.environ.get('MONGODB_API_KEY')
MONGODB_API_APPNAME = os.environ.get('MONGODB_API_APPNAME')

uri = f"mongodb+srv://{MONGODB_API_KEY}@cluster0.t7fr2hb.mongodb.net/?retryWrites=true&w=majority&appName={MONGODB_API_APPNAME}"

# Create a new client and connect to the server
client = MongoClient(uri)


# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
    db = client['chat_doc']
    collection = db['history']
    cursor = collection.find({})
    for document in cursor:
        print(document)
        
except Exception as e:
    print(e)
 
bm25 = BM25Encoder.default() 
sparse_vector = bm25.encode_documents("The brown fox is quick")

encoder = OpenAIEncoder()
dense_vector = encoder.encode_queries(["Who jumped over the lazy dog?"])

hybrid_dense, hybrid_sparse = hybrid_convex_scale(dense_vector[0], sparse_vector, alpha=0.8)
 

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

video_url = "https://www.youtube.com/watch?v=dzmRE4h-FHo&t=22s" 
video_id = extract_youtube_id(video_url)
if video_id:
    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])  
    formatter = TextFormatter()
    formatted_transcript = formatter.format_transcript(transcript)
    tiktoken_encoding = tiktoken.get_encoding("cl100k_base")
    chunks = split_string_with_limit(formatted_transcript, 512,tiktoken_encoding)
    print(chunks[0])