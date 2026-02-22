from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
import os

# Load and chunk PDFs
def load_and_chunk_pdfs(pdf_folder):
    all_chunks = []
    for filename in os.listdir(pdf_folder):
        #print(filename)
        if filename.endswith('.pdf'):
            file_path = os.path.join(pdf_folder, filename)
            #print(file_path)
            loader = PyPDFLoader(file_path)
            document = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )

            for doc in document:
                chunks = text_splitter.split_text(doc.page_content)
                all_chunks.extend(chunks)

    return all_chunks

# Upload chunks to Qdrant
def upload_chunks_to_qdrant(chunks):
    qdrant_client = QdrantClient(url="http://qdrant:6333")

    points = []
    for i, chunk in enumerate(chunks):
        print(i)
        points.append(
            PointStruct(
                id=i,
                payload={"text": chunk},
                vector=[0.0] * 768,  # Replace with actual embeddings
            )
        )

    qdrant_client.upsert(collection_name="pdf_chunks", points=points)

if __name__ == "__main__":
    pdf_folder = "data"  # Your PDF folder path
    chunks = load_and_chunk_pdfs(pdf_folder)
    upload_chunks_to_qdrant(chunks)
    #print(chunks)