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
schema = schema = {
    "customers": {
        "customer_id": "Customer unique ID",
        "name": "Customer full name",
        "dob": "Date of birth",
        "gender": "Gender of the customer",
        "phone_number": "Customer phone number",
        "email": "Customer email address",
        "address": "Customer residential address",
        "city": "City of residence",
        "state": "State of residence",
        "nationality": "Nationality of the customer",
        "created_at": "Timestamp when the customer was created in system"
    },
    "branches": {
        "branch_id": "Branch unique ID",
        "branch_name": "Branch name",
        "city": "City where the branch is located",
        "state": "State where the branch is located",
        "pincode": "Branch pincode",
        "manager_name": "Branch manager's name",
        "contact_number": "Branch contact number",
        "opened_date": "Date when branch was opened",
        "status": "Operational status of the branch",
        "IFSC_code": "Branch IFSC code"
    },
    "casa_accounts": {
        "account_id": "Account unique ID",
        "customer_id": "Reference to the customer",
        "account_number": "Unique CASA account number",
        "account_type": "Type of CASA account (Savings/Current)",
        "branch_id": "Reference to the branch",
        "balance": "Current account balance",
        "currency": "Currency of the account",
        "opened_date": "Date account was opened",
        "status": "Status of the account (Active/Inactive)",
        "last_activity_date": "Date of last transaction activity",
        "interest_rate": "Interest rate applicable to the account"
    },
    "debit_cards": {
        "card_id": "Debit card unique ID",
        "account_id": "Reference to CASA account",
        "card_number": "Unique debit card number",
        "card_type": "Type of debit card (Visa/Mastercard/RuPay)",
        "issued_date": "Date card was issued",
        "expiry_date": "Date card will expire",
        "cvv": "Card CVV code",
        "status": "Status of the card (Active/Blocked)",
        "daily_limit": "Daily transaction limit",
        "monthly_limit": "Monthly transaction limit",
        "branch_id": "Reference to issuing branch"
    },
    "credit_cards": {
        "card_id": "Credit card unique ID",
        "customer_id": "Reference to the customer",
        "card_number": "Unique credit card number",
        "card_type": "Type of credit card (Platinum/Gold/Classic)",
        "credit_limit": "Credit limit assigned to the card",
        "outstanding_amount": "Outstanding credit card dues",
        "issued_date": "Date card was issued",
        "expiry_date": "Date card will expire",
        "status": "Status of the card (Active/Blocked)",
        "rewards_points": "Accumulated reward points",
        "branch_id": "Reference to issuing branch"
    },
    "eop_balances": {
        "balance_id": "Balance record unique ID",
        "account_id": "Reference to CASA account",
        "date": "Date of balance record",
        "opening_balance": "Opening balance of the day",
        "closing_balance": "Closing balance of the day",
        "credits": "Total credits during the day",
        "debits": "Total debits during the day",
        "avg_daily_balance": "Average daily balance",
        "currency": "Currency of the balance",
        "branch_id": "Reference to branch"
    },
    "insurance_policies": {
        "policy_id": "Insurance policy unique ID",
        "customer_id": "Reference to customer",
        "policy_number": "Unique insurance policy number",
        "policy_type": "Type of policy (Life/Health/Auto etc.)",
        "coverage_amount": "Insurance coverage amount",
        "premium_amount": "Premium to be paid",
        "start_date": "Policy start date",
        "end_date": "Policy end date",
        "status": "Status of the policy (Active/Expired)",
        "nominee_name": "Name of the policy nominee",
        "agent_id": "Reference to insurance agent"
    },
    "loan_accounts": {
        "loan_id": "Loan unique ID",
        "customer_id": "Reference to customer",
        "loan_account_number": "Unique loan account number",
        "loan_type": "Type of loan (Home/Auto/Personal)",
        "principal_amount": "Principal loan amount sanctioned",
        "outstanding_amount": "Current outstanding loan amount",
        "interest_rate": "Interest rate of the loan",
        "tenure_months": "Loan tenure in months",
        "start_date": "Loan start date",
        "end_date": "Loan end date",
        "status": "Status of the loan (Active/Closed)",
        "branch_id": "Reference to branch"
    },
    "loan_accounts_history": {
        "history_id": "History record unique ID",
        "loan_id": "Reference to loan account",
        "date": "Date of history record",
        "principal_outstanding": "Principal outstanding on that date",
        "interest_outstanding": "Interest outstanding on that date",
        "total_due": "Total due amount on that date",
        "payment_made": "Payment made on that date",
        "overdue_amount": "Overdue amount if any",
        "status": "Status of the loan on that date",
        "remarks": "Additional remarks or notes"
    },
    "transactions": {
        "transaction_id": "Transaction unique ID",
        "account_id": "Reference to CASA account",
        "transaction_date": "Date and time of transaction",
        "transaction_type": "Type of transaction (Credit/Debit)",
        "amount": "Transaction amount",
        "balance_after": "Balance after transaction",
        "description": "Transaction description",
        "channel": "Channel used (ATM, Online, Branch)",
        "reference_number": "Unique reference number",
        "status": "Status of the transaction (Success/Failed)",
        "branch_id": "Reference to branch"
    }
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

    print("Pruned schema:", pruned_schema)

    # Structured intent: {structured}
    
    system_prompt = f"""
    You are an expert SQL generator.
    User query: "{masked_query}"
    Relevant schema: {pruned_schema}

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
        
    # Final Step: Generate human readable insights from the results and the query
    system_prompt = f"""
    You are an expert business and data analyst who excels at deriving insights from data.
    Given the user's query, the SQL executed, and the results obtained, provide a concise and
    relevant insight that directly addresses the user's original question.
    If the results are empty or do not provide any insight, respond with "No insights available."
    User query: "{masked_query}"
    Results: {rows}
    
    Provide the insights in the form of markdown. Use tables if needed.
    """
    
    insights = openai.responses.create(
        model="gpt-5-mini",
        input=system_prompt,
    ).output_text.strip()
    
    print("Insights:", insights)

    return {"sql": unmasked_sql, "result": rows, "masked_sql": sql, "insights": insights}
