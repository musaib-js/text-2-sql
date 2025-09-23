import spacy
import openai
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import os
load_dotenv()

nlp = spacy.load("en_core_web_sm")


openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()


class QueryRequest(BaseModel):
    query: str


# --- Semantic search helper ---
def get_embedding(text: str) -> list[float]:
    """Fetch embedding from OpenAI API"""
    resp = openai.embeddings.create(model="text-embedding-3-small", input=text)
    return resp.data[0].embedding


def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def semantic_match(query_term: str, schema: dict):
    query_emb = get_embedding(query_term)
    best_table, best_col, best_score = None, None, -1

    for table, cols in schema.items():
        for col, desc in cols.items():
            col_emb = get_embedding(desc)
            score = cosine_similarity(query_emb, col_emb)
            if score > best_score:
                best_table, best_col, best_score = table, col, score

    return best_table, best_col


@app.post("/ask")
def ask(req: QueryRequest):
    query = req.query

    # Step 1: NER for gettin entities
    doc = nlp(query)
    entities = [ent.text for ent in doc.ents]

    # Step 2: Schema metadata
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

    # Step 3: Semantic matching
    select_col = semantic_match(query, schema)

    filter_value = entities[0] if entities else None
    filter_col = None
    if filter_value:
        filter_col = semantic_match(filter_value, schema)

    structured = {
        "table": "customers",
        "select": [select_col],
        "filter": {filter_col: filter_value} if filter_col else None,
    }
    
    print("Structured intent:", structured)

    # Step 4: LLM Based SQL generation
    system_prompt = f"""
    User query: "{query}"
    Schema: {schema}
    Structured intent: {structured}
    Generate valid SQL for Postgres.
    """

    sql = (
        openai.responses.create(
            model="gpt-5-mini",
            input=system_prompt
        )
    )
    
    sql = sql.output_text.strip()

    return {"sql": sql}
