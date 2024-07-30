from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_openai     import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
import os
import shutil
import pandas as pd
import re
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from openai import OpenAI
import json

client = OpenAI(api_key="<insert_open_ai_key>")




indexepath = '/storage/david/documenti/phd/articoli_analizzati/Articoli_e_vos_Viewer/scopus_input.csv'
outfilename = '/storage/david/documenti/phd/articoli_analizzati/Articoli_e_vos_Viewer/scopus_output.csv'

df = pd.read_csv(indexepath)

df['Compliant'] = ""
df['Reason'] = ""


PROMPT_TEMPLATE = """
The following values for Title and Abstract should describe a study regarding the prediction of student drop out in bachelor or masters courses, using artificial intelligence and: without being mostly theoretical; not involving remote classes or courses; not being focussed on improving teaching or evaluation; not being focussed on improving the university environment.
Prepare a response in the form {"compliant":"article_compliancy","reason":"non_compliancy_reason"}.
The value of article_compliancy should be YES is the abstract really describes such kind of study, otherwise the value of article_compliancy should be NO. the value of "non_compliancy_reason" if article_compliancy is NO should contain the main reason for which the abstract does not comply. If article_compliancy is YES then non_compliancy_reason should only have the value "Complies"
"""


def get_series(eid):
    in_record = df.loc[df['EID'] == eid].iloc[0]
    return in_record



def main():
    i=1

    for index, row in df.iterrows():
        eid=row['EID']
        title_text=row['Title']
        abstract_text=row['Abstract']

        CONTENT_TEMPLATE = f"the input is title: {title_text} and abstract {abstract_text}."

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": PROMPT_TEMPLATE},
                {"role": "user", "content": CONTENT_TEMPLATE}
            ]
        )

        answer = response.choices[0].message.content
        answer = answer.replace('```json','').replace('```','')
        try:
            data = json.loads(answer)
            df.at[index,'Compliant'] = data['compliant']
            df.at[index,'Reason'] = data['reason']
            print('OK')
            print(str(index)+" "+eid)
            print(response.choices[0].message.content)
        except:
            print('KO')
            print(str(index)+" "+eid)
            print(response.choices[0].message.content)
            break

        i=i+1
        if i> 20:
            break

    df.to_csv(outfilename, index=False)



if __name__ == "__main__":
    main()

