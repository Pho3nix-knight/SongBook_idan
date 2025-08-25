import os
from dotenv import load_dotenv

# === CONFIGURABLE VARIABLES ===
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHUNKS_PATH = "./songbook_chunks.jsonl"
FILE_PATH = "./songbook.txt"
CHROMA_DIR = "./songbook_chroma"
RAGAS_QNA_EVALUATION_PATH = "./ragas_evaluations/ragas_eval_qna_result.txt"
