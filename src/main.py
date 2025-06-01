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
import json
import dotenv

# Load API keys as environmental variables
dotenv.load_dotenv()

def load_lottie_file(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)


with st.sidebar:
    st.title("**PodcastAI.**")
    st.button(":heavy_plus_sign: New Chat")


def strip_think_sections(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

team = agents.get_agent_team()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("PodcastAI.")

# Display chat history
if not st.session_state.messages:
    st.info("Start chatting.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Enter a podcast idea..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        placeholder = st.empty()
        cursor_placeholder = st.empty()
        
        with cursor_placeholder:
            st_lottie(load_lottie_file("assets/lotte-loading.json"), height=60)

        # Step 1: Capture full response
        response: Iterator[RunResponse] = team.run(prompt, stream=True, session_id="session")
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
