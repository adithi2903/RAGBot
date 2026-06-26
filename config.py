"""Central configuration + the Nemotron client. Imported by the other files."""
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # reads a .env file in this folder (see .env.example)

NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    raise RuntimeError(
        "NVIDIA_API_KEY not set. Copy .env.example to .env and put your key in it. "
        "Get a free key at https://build.nvidia.com"
    )

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY,
    timeout=120,
)

# Develop on the faster Super model; switch to Ultra for the final demo.
MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1.5"
# MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1.5"

# Paths
PDF_DIR = "tax_pdfs"        # put your renamed PDFs here
CHROMA_DIR = "chroma_db"    # the vector database is saved here

# Models
EMB_MODEL = "BAAI/bge-small-en-v1.5"                 # embeddings
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # reranker

# Retrieval / chunking knobs
COLLECTION = "tax"
CHUNK_SIZE = 1500
OVERLAP = 200
POOL = 40      # chunks pulled before reranking
TOP_K = 5      # chunks kept after reranking