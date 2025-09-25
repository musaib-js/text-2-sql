import os
import spacy
import openai
import redis
import numpy as np
import psycopg2
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from pii_masking import PIIMasker
from cryptography.fernet import Fernet

load_dotenv()

# --- Setup ---
nlp = spacy.load("en_core_web_sm")

openai.api_key = os.getenv("OPENAI_API_KEY")

redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST"),
    port=int(os.getenv("REDIS_PORT")),
    decode_responses=True,
)

db_conn = psycopg2.connect(
    dbname=os.getenv("POSTGRES_DB"),
    user=os.getenv("POSTGRES_USER"),
    password=os.getenv("POSTGRES_PASSWORD"),
    host=os.getenv("POSTGRES_HOST"),
)

app = FastAPI()


class QueryRequest(BaseModel):
    query: str


# --- Embedding + Cache ---
def get_embedding(text: str) -> list[float]:
    key = f"emb:{text}"
    cached = redis_client.get(key)
    if cached:
        return np.fromstring(cached, sep=",").tolist()

    resp = openai.embeddings.create(model="text-embedding-3-small", input=text)
    emb = resp.data[0].embedding
    redis_client.set(key, ",".join(map(str, emb)))
    return emb


def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# --- Schema Metadata ---
schema = {
    "customers": {
        "id": "Customer unique ID",
        "name": "Customer full name",
        "phone_number": "Customer's phone number",
        "email": "Customer email address",
    },
    "orders": {
        "id": "Order unique ID",
        "customer_id": "Reference to the customer",
        "product_id": "Reference to the product",
        "order_date": "Date of the order",
        "amount": "Order total amount",
    },
    "products": {
        "id": "Product unique ID",
        "name": "Product name",
        "category": "Product category",
        "price": "Product price",
    },
    "employees": {
        "id": "Employee unique ID",
        "name": "Employee full name",
        "position": "Employee job position",
        "store_id": "Reference to the store",
    },
    "stores": {
        "id": "Store unique ID",
        "name": "Store name",
        "city": "City where the store is located",
    },
}


def semantic_match(query_term: str, schema: dict, top_k: int = 2):
    query_emb = get_embedding(query_term)
    scored = []
    for table, cols in schema.items():
        for col, desc in cols.items():
            col_emb = get_embedding(desc)
            score = cosine_similarity(query_emb, col_emb)
            scored.append((table, col, score))
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored[:top_k]


@app.post("/ask")
def ask(req: QueryRequest):
    query = req.query
    doc = nlp(query)
    entities = [ent.text for ent in doc.ents]

    select_candidates = semantic_match(query, schema, top_k=2)
    filter_candidates = []
    if entities:
        for ent in entities:
            filter_candidates.extend(semantic_match(ent, schema, top_k=1))

    relevant_tables = list(
        set([t for (t, _, _) in select_candidates + filter_candidates])
    )
    pruned_schema = {t: schema[t] for t in relevant_tables}
    
    print("Relevant tables:", relevant_tables)
    print("Select candidates:", select_candidates)
    print("Filter candidates:", filter_candidates)
    print("Pruned schema:", pruned_schema)

    structured = {
        "tables": relevant_tables,
        "select_candidates": [(t, c) for (t, c, _) in select_candidates],
        "filters": [
            {"entity": ent, "candidate": (t, c)}
            for ent, (t, c, _) in zip(entities, filter_candidates)
        ],
    }
    
    print("Structured intent:", structured) 
    
    # --- PII Masking ---
    
    # Step 1 --> Mask the query

    pii_masker = PIIMasker(nlp, secret_key=os.getenv("FERNET_KEY", Fernet.generate_key().decode()))
    
    masked_query = pii_masker.mask(query)
    
    print("Masked query:", masked_query)
    
    
    # Step 2 --> Mask the structured intent
    for filter in structured["filters"]:
        original = filter["entity"]
        masked = pii_masker.mask(original)
        filter["entity"] = masked
        
    print("Masked structured intent:", structured)
    
    system_prompt = f"""
    You are an expert SQL generator.
    User query: "{masked_query}"
    Relevant schema: {pruned_schema}
    Structured intent: {structured}

    Generate valid SQL for Postgres.
    
    Output only the SQL without any explanation.
    """

    sql = openai.responses.create(
        model="gpt-5-mini",
        input=system_prompt,
    ).output_text.strip()
    
    print("Generated SQL (masked):", sql)
    
    # Step 3 --> Unmask the SQL
    unmasked_sql = pii_masker.unmask(sql)
    
    print("Unmasked SQL:", unmasked_sql)


    rows = []
    
    db_conn.autocommit = True  # for read-only workloads

    try:
        with db_conn.cursor() as cursor:
            cursor.execute(unmasked_sql)
            rows = cursor.fetchall()
    except Exception as e:
        print("SQL Execution Error:", e)
        rows = []

    return {"sql": unmasked_sql, "result": rows, "masked_sql": sql}
