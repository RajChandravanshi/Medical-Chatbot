from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings


# Extract text from pdf files
def load_pdf_files(data):
    loader = DirectoryLoader(
        data,
        glob = "*.pdf",
        loader_cls = PyPDFLoader
    )
    documents = loader.load()
    return documents

def filter_to_minimal_docs(docs : List[Document]) -> List[Document]:
    """
    Gives a list of document objects, return a new list of document objects containing 
    only 'source' in metadata and the original page_content.
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get('source')
        minimal_docs.append(
            Document(
                page_content = doc.page_content,
                metadata = {
                    'source':src
                }
            )
        )

    return minimal_docs

# chunking
def text_split(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 20
    )
    texts_chunk = text_splitter.split_documents(docs)
    return texts_chunk


# downloading the embedding model

def download_embeddings_model():
    """
    Download the embeddings model 'sentence-transformers/all-MiniLM-L6-v2'
    for converting text into numerical vectors.
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name
    )
    return embeddings   

embedding = download_embeddings_model()