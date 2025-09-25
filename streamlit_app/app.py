import os
import requests
import streamlit as st

st.set_page_config(page_title="Banking Insights", layout="wide")
st.title("🏦 Banking Data Insights")

API_URL = os.getenv("API_URL", "http://localhost:8000")

query = st.text_input(
    "Enter your query",
    placeholder="e.g., Show the average loan outstanding amount by branch",
)

if st.button("Submit"):
    try:
        resp = requests.post(f"{API_URL}/ask", json={"query": query})
        if resp.status_code == 200:
            data = resp.json()
            st.markdown(data.get("insights", "No insights available"))
        else:
            st.error(f"API Error: {resp.text}")
    except Exception as e:
        st.error(f"Failed to connect to API: {e}")
