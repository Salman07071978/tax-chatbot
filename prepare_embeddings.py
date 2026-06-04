from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np

print("Loading PDF...")

reader = PdfReader("taxlaw.pdf")

text = ""
for page in reader.pages:
    page_text = page.extract_text()
    if page_text:
        text += page_text

print("PDF length:", len(text))

if len(text) < 100:
    raise Exception("PDF not loaded properly")

# Split text
chunks = [text[i:i+800] for i in range(0, len(text), 800)]

print("Chunks created:", len(chunks))

# Model
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Creating embeddings...")

embeddings = model.encode(chunks, show_progress_bar=True)

print("Saving file...")

np.savez("tax_data.npz", chunks=chunks, embeddings=embeddings)

print("DONE ✔ tax_data.npz created successfully")
