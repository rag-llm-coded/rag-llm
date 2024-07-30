#### This file will run a query against a specially trained database made from pdf files of academic interest
#### This process i called RAG, i.e. Retrieval Augmented Generation.
#### The creation of vectors as index of text chunks is called "Embedding"
#### This file uses openai as the LLM but the process is possible with other large language models albeit the libraries used and functions called will be different.
#### Since the documents are of academic interest each "embedding" (i.e. vector) have received added metadata extracted from Scopus when added to the database
#### The steps that will be followed are:
####
##### user runs the following command
##### python query_data_generic.py "my question?"
##### where "my question?" is the question the user desires to be answered
##### the query is embedded like the pdf training set
##### the chunks of text in the chroma database are compared to the embedded version of the query using the function:similarity_search_with_relevance_scores
##### the 40 chunks that match the question best are returned together with each chunk's score.
##### The number of chunks returned can be changed modifying the parameter k=40
##### if no positive score is retrieved then the script prints "Unable to find matching results" and exits
##### if some positive and some negative scores are retrieved the chunks with negative scores are removed - Keeping negative scores could give a correct answer but it could cause the LLM to hallucinate
##### The remaining chunks are concatenated into a string called "Context"
##### A csv file is produced called "my_question_.csv" in the subditectory ./csv of where the script is run (needs to be present).
##### The name of the csv file is taken from the text of the query substituting spaces and punctuation with the character '_'
##### An LLM (in this case openai using ChatGpt4o) is invoked asking to find an answer to the query based exclusively on the Context which is passed together with the query.
##### The csv file is created compling all the Context chunks and the associated metadata
##### After the context chunks the query and LLM response are added and the file is closed
##### The Context chunks and LLM response are also shown on the console.
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

#### This script also assumes that from the directory where it is run there is a subdirectory ./csv
##### if it is not present create the directory before running the script



import argparse
from dataclasses import dataclass
from langchain.vectorstores.chroma import Chroma
from langchain_openai     import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import pandas as pd
import time
import string
import warnings



# Suppress all warnings
warnings.simplefilter("ignore")

# Insert here the path to the directory where the chroma, i.e. sqlite files are
CHROMA_PATH = "/<path to chroma>/chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Prepare the DB.
    # The model used here is text-embedding-3-large but it's not a requirement. What is required is that if training is done with the model <my_model> then the query script must use the same
    embedding_function = OpenAIEmbeddings(model="text-embedding-3-large")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=40)
    if len(results) == 0 or results[0][1] < 0:
        print(f"Unable to find matching results.")
        return

    subresults=[(doc,score) for doc, score in results if score > 0]


    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in subresults])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    model = ChatOpenAI()
    response_text = model.predict(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in subresults]
    formatted_response = f"Response: {response_text}"
    print(formatted_response)


    my_docs = [doc[0] for doc in subresults]
    meta_doc = [doc.metadata for doc in my_docs]
    df = pd.DataFrame(meta_doc)


    dfr = df[['EID','Year','Title', 'DOI', 'Authors','Source title','page']].copy()
    dfr['page']=dfr['page'].astype(int)+1


    scores=[score for doc,score in subresults]
    dfr ['scores'] = scores

    page_contents= [doc.page_content for doc in my_docs]
    dfr ['page_contents'] = page_contents


    dfr.loc[len(dfr)]= ['------',None,'------','------','------','------','------',None,None]
    dfr.loc[len(dfr)]= ['Query',None,None,None,None,None,None,None,query_text]
    dfr.loc[len(dfr)]= ['------',None,'------','------','------','------','------',None,None]
    dfr.loc[len(dfr)]= ['Response',None,None,None,None,None,None,None,response_text]


    translation_table = str.maketrans(string.punctuation, '_' * len(string.punctuation))
    outfilename='./csv/'+query_text.translate(translation_table).replace(' ','_')+".csv"


    #outfilename=str(int(time.time()))+".csv"
    dfr.to_csv(outfilename, index=False)



if __name__ == "__main__":
    main()
