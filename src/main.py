import agents

from typing import Iterator
from agno.agent import RunResponse
from textwrap import dedent
from agno.utils.pprint import pprint_run_response

import streamlit as st
from streamlit_lottie import st_lottie
import pandas as pd
import numpy as np
import time
import re
import os
import json
import dotenv
import secrets

# Load API keys as environmental variables
dotenv.load_dotenv()
conversations_file = "temp/conversations.json"

# Helper functions: =============================================================================
def load_json(filepath: str):
    with open(filepath, 'r') as f:
        return json.load(f)

def strip_think_sections(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

def get_session_id():
    return str(secrets.randbits(64))
# ===============================================================================================

EMPTY_SESSIONS_DATA = { "sessions": [] }

if not os.path.exists(conversations_file):
    with open(conversations_file, 'w') as f:
        json.dump(EMPTY_SESSIONS_DATA, f)
   

with open(conversations_file, 'r') as f: 
    session_conversation_data = json.load(f)
    if not session_conversation_data["sessions"]:
        current_session_id = get_session_id()
        session_conversation_data["sessions"] = [{"session_id": current_session_id, "messages": []}]
        current_session_messages = session_conversation_data.get("sessions")[0]["messages"]
    else:
        current_session_id = session_conversation_data["sessions"][0]["session_id"]
        if not "messages" in session_conversation_data["sessions"][0]:
            session_conversation_data["sessions"][0]["messages"] = []
        current_session_messages = session_conversation_data.get("sessions")[0]["messages"]

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = current_session_messages

team = agents.get_agent_team()
# current_session_messages = session_conversation_data.get("sessions")[0]

# UI =============================================================================================
with st.sidebar:
    st.title("**PodcastAI.**")
    st.button(":heavy_plus_sign: New Chat")

st.title("PodcastAI.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Enter a podcast idea..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    current_session_messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        placeholder = st.empty()
        cursor_placeholder = st.empty()
        
        with cursor_placeholder:
            st_lottie(load_json("assets/lotte-loading.json"), height=60)

        response: Iterator[RunResponse] = team.run(prompt, stream=True)
        full_text = ""
        for chunk in response:
            if hasattr(chunk, 'content') and chunk.content:
                full_text += chunk.content

        full_text = strip_think_sections(full_text)
        cursor_placeholder.empty()

        markdown_text = ""
        for char in full_text:
            markdown_text += char
            placeholder.markdown(markdown_text)
            time.sleep(0.005)  # Adjust for speed
        st.session_state.messages.append({"role": "assistant", "content": full_text})
        current_session_messages.append({"role": "assistant", "content": full_text})

    with open(conversations_file, "w") as f:
        json.dump(session_conversation_data, f, indent=4)

print(current_session_id)
