#### This file will train a set of pdf documents of Academic interest against a LLM and transform chunks of the pdf files into vectors as to be able to search and retrieve at a later time.
#### This process is called RAG, i.e. Retrieval Augmented Generation.
#### The creation of vectors as index of text chunks is called "Embedding"
#### This file uses openai as the LLM but the process is possible with other large language models albeit the libraries used and functions called will be different.
#### Since the documents are of academic interest each "embedding" (i.e. vector) will receive added metadata extracted from Scopus
#### The steps that will be followed are:
##### Parse the pdf documents and extract text
##### Add Scopus metadata to the documents
##### Split the documents into chunks copying the metadata of each document all of its chunks
##### Create a Chroma Database
##### Execute embedding and save data and metadata to the Chroma Database
##### The code was inspired from the following Git repositories: https://github.com/pixegami
##### The following video by the same author was also of inspiration https://www.youtube.com/watch?v=2TJxpyO3ei4

#### Remember Before running install the following
#### pip install langchain-community langchain-openai langchain pandas unstructured unstructured[pdf] chromadb openai

#### This script assumes that at an operating system level a system variable called OPENAI_API_KEY has been set.
#### on linux this is done running
#####    export OPENAI_API_KEY='<my Open Ai key>'
####
#### The same result can be achieved by adding the following line at the start of the script
##### os.environ["OPENAI_API_KEY"] = "<my Open Ai key>"


from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_openai     import OpenAIEmbeddings
import os
import shutil
import pandas as pd
import re

# Insert here the path to the directory where you want the chroma, i.e. sqlite files be created
CHROMA_PATH = "/<path to chroma>/chroma"

# Insert here the base path to the directory where the documents to be used for training are stored
# -- The documents can be stored in structured subdirectories. All will be traversed
# -- File names must start with the first 18 characters corresponding to the Scopus EID code that is exported with any Scopus Search by default.
DATA_PATH = "/<base path to training documents>"


# Insert here an export of the documents metadata taken from Scopus, hereafter referred to as Index. The index may be bigger than the list of available documents but must include all documents
# -- If a document is missing from the index it will still be used for the training but the metadata will not be associated with it hence queries will not yield the appropriate references.
#
# The index can be extracted from scopus.com accessed with appropriate credentials that allow the user do perform and export searches, with the following steps:
# -- 1 Run a search on Scopus that returns the list of documents of interest. How the search is performed is unimportant as long as the number returned is not greater than 20000.
# -- 2 The search will yield a number of results and the returning page will have a clear indication with "XXX documents found"
# -- 3 Right below there is a link to "Export" that must be clicked.
# -- 4 In the modal dialog choose CSV.
# -- 5 Another modal dialog opens and select as follows:
# -----5.1 Choose to select "Documents from 1 to XXX"
# -----5.2 In the lower section where the information to be exported is:
# -------5.2.1. Keep the selected items (Citation Information)
# -------5.2.2. Add all items in the section "Abstract and Keywords"
# -------5.2.3. Add "Include References" from the section "Other Information"
# -----5.3 press Export and save the file to some directory that will be now on called <path to Scopus Index>


indexepath = '/<path to Scopus Index>/scopus_input.csv'
df = pd.read_csv(indexepath)



def get_series(eid):
    in_record = df.loc[df['EID'] == eid].iloc[0]
    return in_record

def add_metadata(document):
    file_path = document.metadata['source']
    filename = os.path.basename(file_path)
    filekey = filename[0:18]
    record = get_series(filekey)
    document.metadata['Authors'] =record['Authors']
    document.metadata['Author full names'] =record['Author full names']
    document.metadata['Author(s) ID'] =record['Author(s) ID']
    document.metadata['Title'] =record['Title']
    document.metadata['Year'] =record['Year'].astype(int).tolist()
    document.metadata['Source title'] =record['Source title']
    document.metadata['Volume'] =record['Volume']
    document.metadata['Issue'] =record['Issue']
    document.metadata['Art. No.'] =record['Art. No.']
    document.metadata['Page start'] =record['Page start']
    document.metadata['Page end'] =record['Page end']
    document.metadata['Page count'] =record['Page count']
    document.metadata['Cited by'] =record['Cited by'].astype(int).tolist()
    document.metadata['DOI'] =record['DOI']
    document.metadata['Link'] =record['Link']
    document.metadata['Affiliations'] =record['Affiliations']
    document.metadata['Authors with affiliations'] =record['Authors with affiliations']
    document.metadata['Abstract'] =record['Abstract']
    document.metadata['Author Keywords'] =record['Author Keywords']
    document.metadata['Index Keywords'] =record['Index Keywords']
    document.metadata['References'] =record['References']
    document.metadata['Correspondence Address'] =record['Correspondence Address']
    document.metadata['Document Type'] =record['Document Type']
    document.metadata['Publication Stage'] =record['Publication Stage']
    document.metadata['Open Access'] =record['Open Access']
    document.metadata['Source'] =record['Source']
    document.metadata['EID'] =record['EID']



def main():
    generate_data_store()

def generate_data_store():
    documents = load_documents()
    for document in documents:
        add_metadata(document)
    chunks = split_documents(documents)
    save_to_chroma(chunks)


def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents.
    # The model used here is text-embedding-3-large but it's not a requirement. What is required is that if training is done with the model <my_model> then the query script must use the same
    db = Chroma.from_documents(
        chunks,
        OpenAIEmbeddings(model="text-embedding-3-large"),
        persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

if __name__ == "__main__":
    main()

