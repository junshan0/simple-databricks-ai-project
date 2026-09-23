# app.py (Streamlit sketch)
import json
import streamlit as st
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
ENDPOINT = "agents_video_ai-ai-video_rag_agent"

st.title("Video RAG Chatbot")

q = st.chat_input("Ask about the videos…")
if q:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        resp = w.api_client.do(
            "POST",
            f"/serving-endpoints/{ENDPOINT}/invocations",
            body={"messages": [{"role": "user", "content": q}]},
            raw=True,
        )
        raw = resp["contents"].read()
        data = json.loads(raw)
        answer = data["messages"][-1]["content"]
        st.write(answer)