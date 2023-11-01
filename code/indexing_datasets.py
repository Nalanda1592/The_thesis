import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import pinecone
from tqdm.auto import tqdm
from transformers import pipeline, TapasTokenizer, TapasForQuestionAnswering

def preprocess_table(data: pd.DataFrame):
    processed = ["Gender_Female,Gender_Male,Education_Masters,Education_PhD,Age,Experience,Interview_Score,Test_Score,Actual Salary,PredictedSalary\n"]

    x = data.to_string(header=False,index=False,index_names=False).split('\n')
    for ele in x:
        record = ','.join(ele.split())
        #record =', '.join([f"'{w}'" for w in ele.split()])
        processed.append(record+"\n")

    print(processed[2])
    return processed

def store_vector(processed_table,index,retriever: SentenceTransformer):
    # we will use batches of 64
    batch_size = 300
    for i in tqdm(range(0, len(processed_table), batch_size)):
        # find end of batch
        i_end = min(i+batch_size, len(processed_table))
        # extract batch
        batch = processed_table[i:i_end]
        # generate embeddings for batch
        emb = retriever.encode(batch).tolist()
        # create unique IDs ranging from zero to the total number of tables in the dataset
        ids = [f"{idx}" for idx in range(i, i_end)]
        # add all to upsert list
        to_upsert = list(zip(ids, emb))
        # upsert/insert these records to pinecone
        done = index.upsert(vectors=to_upsert)
    print("done2")

def query_pinecone(query,index,processed_table,retriever: SentenceTransformer):
    # generate embedding for the query
    xq = retriever.encode([query]).tolist()
    # query pinecone index to find the table containing answer to the query
    top_tables=[]
    result = index.query(xq, top_k=5)
    # return the relevant tables from the tables list
    for i in range(1,5):
        raw=processed_table[int(result["matches"][i]["id"])]
        new=raw[0:-1]
        input_list=new.split(",")
        top_tables.append(input_list)
    dataframe_table=pd.DataFrame(top_tables,columns=['Gender_Female', 'Gender_Male', 'Education_Masters', 'Education_PhD', 'Age', 'Experience', 'Interview_Score', 'Test_Score','Actual Salary','PredictedSalary'])
    print(dataframe_table)
    return dataframe_table

def get_answer_from_table(processed_table: pd.DataFrame, pipe: pipeline,query: str):
    # run the table and query through the question-answering pipeline
    answers = pipe(table=processed_table, query=query)
    #pipe(table=processed_table, query=query)
    return answers

def main():

    data = None

    current_dir = os.getcwd()
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

    data = pd.read_csv(current_dir + "/data/total_prediction_results/Datensatz1_results/" + "total_database_prediction.csv")
    print("done1")

    # load the table embedding model from huggingface models hub
    retriever = SentenceTransformer("deepset/all-mpnet-base-v2-table", device='cpu')

    processed_table = preprocess_table(data)
    print(processed_table[1])

    pinecone.init(api_key="636de315-26de-4439-bdf1-5603788c6963", environment="gcp-starter")
    index_name= "explainer-index"

    index = pinecone.Index(index_name)

    query_result = retriever.encode("Hello world").tolist()
    print(len(query_result))

    #store_vector(processed_table,index,retriever)

    # check that we have all vectors in index
    print(index.describe_index_stats())

    model_name = "google/tapas-large-finetuned-wtq"
    # load the tokenizer and the model from huggingface model hub
    tokenizer = TapasTokenizer.from_pretrained(model_name)
    model = TapasForQuestionAnswering.from_pretrained(model_name, local_files_only=False)
    # load the model and tokenizer into a question-answering pipeline
    pipe = pipeline(task="table-question-answering",  model=model, tokenizer=tokenizer, device='cpu')
    query = "what is the lowest actual salary in database?"
    

    str_data=data.applymap(str)

    if(query.__contains__('database')):
        answer=get_answer_from_table(str_data, pipe, query)
        print(answer['answer'])
    else:
        table = query_pinecone(query,index,processed_table,retriever)
        answer=get_answer_from_table(table, pipe, query)
        print(answer['answer'])


if __name__=='__main__':
    main()