import logging
import os
import chromadb
from tqdm import tqdm
from typing import List, Any
import gc
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from chromadb import Client
from chromadb.config import Settings
import gradio as gr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QABot:
    """
    QABot is a class that sets up a Question-Answering bot using multiple PDF files as context.
    It combines Hugging Face models for language processing and Sentence Transformers for 
    embedding and document retrieval.
    """
    
    def __init__(self) -> None:
        self.llm_model_name = 'google/flan-t5-large'
        self.embedding_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
        self.top_k_docs = 4
        self.max_lenght = 512
        self.max_new_tokens = 1000
        self.chunk_size = 400
        self.chunk_overlap = 50
        self.temperature = 0.4
        self.task = 'text2text-generation'

    def get_llm(self):
        """Initialize the LLM for QA tasks."""
        logger.info(f"Loading LLM model: {self.llm_model_name}")

        tokenizer = AutoTokenizer.from_pretrained(
            self.llm_model_name, 
            padding=True, 
            truncation=True,
            return_tensors="pt"
        )

        llm_pipeline = pipeline(
            task=self.task,
            model=self.llm_model_name, 
            tokenizer=tokenizer,
            model_kwargs={
                'do_sample': True,
                'temperature': self.temperature 
            },
            max_new_tokens=self.max_new_tokens
        )
        llm = HuggingFacePipeline(pipeline=llm_pipeline)
        return llm
   
    def get_embedding_model(self):
        """Initialize the embedding model."""
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        hf_embedding = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        return hf_embedding

    def pdf_document_loader(self, files: List[gr.File]) -> List[Document]:
        """Load text from each PDF file and return as a list of Document objects."""
        logger.info("Loading and processing PDF files...")
        
        documents = []
        for file in tqdm(files, desc="Loading PDFs"):
            try:
                loader = PyPDFLoader(file.name)
                file_documents = loader.load() 
                documents.extend(file_documents)
                logger.info(f"Loaded {len(file_documents)} pages from {file.name}.")
            except Exception as e:
                logger.error(f"Error loading file {file.name}: {e}")
        return documents

    def text_splitter(self, documents: List[Document]) -> List[Document]:
        """Split pdf documents into chunks, retaining the Document structure."""
        logger.info("Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap, 
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Total chunks created: {len(chunks)}")
        gc.collect()
        return chunks
    
    def vector_database(self, chunks):
        "Embed and store chunks in a ChromaDB vector store"
        embedding_model = self.get_embedding_model()
        vectordb = Chroma.from_documents(documents=chunks, embedding=embedding_model)
        return vectordb

    def retriever(self, files: List[gr.File]) -> Any:
        "Define a vector store-based retriever that retrieves information using a simple similarity search"
        splits = self.pdf_document_loader(files)
        chunks = self.text_splitter(splits)
        vectordb = self.vector_database(chunks)
        retriever = vectordb.as_retriever(search_kwargs={'k': self.top_k_docs})
        return retriever
    
    def retriever_qa(self, files: List[gr.File], query: str) -> str:
        """Perform QA using multiple PDF files as context."""
        llm = self.get_llm()
        retriever = self.retriever(files)     
        qa_chain = RetrievalQA.from_chain_type(
                llm=llm, 
                chain_type="refine", 
                retriever=retriever, 
                return_source_documents=False          
            )
        
        # retrivied_docs = retriever.invoke(query)
        # context = " ".join([doc.page_content for doc in retrivied_docs[:self.top_k_docs]])
        # input_data = f"Query : {query}, Context: {context}"
        # response = llm.invoke(input=input_data)
        # print("response: ", response)
        # return response

        try:
            response = qa_chain.invoke(query)
            print('Response: ', response)
            return response['result']
        except Exception as e:
            logger.error(f"Error during QA: {e}")
            return "An error occurred while processing your request."


def main():
        
    def qa_with_error_handling(files, query): 
        try: 
            bot = QABot() 
            return bot.retriever_qa(files, query) 
        except Exception as e: 
            logger.error(f"Error during QA: {e}") 
            return "An error occurred while processing your request."
    
    rag_application = gr.Interface(
        fn=qa_with_error_handling,
        flagging_mode="never",
        inputs=[
            gr.File(label="Upload PDF Files", file_count="multiple", file_types=['.pdf'], type="filepath"),  
            gr.Textbox(label="Input Query", lines=3, placeholder="Type your question here...")
        ],
        outputs=gr.Textbox(label="Output"),
        title="RAG Chatbot with Langchain and Huggingface",
        description="Upload multiple PDF documents and ask any question. The chatbot will try to answer using the provided documents."
    )
    rag_application.launch(server_name="0.0.0.0", server_port=8000)

if __name__ == '__main__':
    main()
