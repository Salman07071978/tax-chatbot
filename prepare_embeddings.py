from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np

# Load PDF
reader = PdfReader("taxlaw.pdf")
text = ""
for page in reader.pages:
    text += page.extract_text() or ""

# Split
chunks = [text[i:i+800] for i in range(0, len(text), 800)]

# Model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Embeddings (ONE TIME ONLY)
embeddings = model.encode(chunks)

# Save to file
np.save("chunks.npy", np.array(chunks, dtype=object))
np.save("embeddings.npy", embeddings)

print("DONE - embeddings saved")
