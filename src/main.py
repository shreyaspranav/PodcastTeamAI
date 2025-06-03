import agents

from typing import Iterator
from agno.agent import RunResponse
from textwrap import dedent
from agno.utils.pprint import pprint_run_response

from agno.memory.v2.memory import Memory
from agno.storage.sqlite import SqliteStorage

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

# Helper functions:
def load_json(filepath: str):
    with open(filepath, 'r') as f:
        return json.load(f)

def strip_think_sections(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

def get_session_id():
    return str(secrets.randbits(64))

EMPTY_SESSIONS_DATA = {"sessions": []}

# Ensure directory exists
os.makedirs(os.path.dirname(conversations_file), exist_ok=True)

# Load or initialize sessions data
if not os.path.exists(conversations_file):
    with open(conversations_file, 'w') as f:
        json.dump(EMPTY_SESSIONS_DATA, f)

with open(conversations_file, 'r') as f:
    session_conversation_data = json.load(f)
    # If no sessions exist, create one
    if not session_conversation_data.get("sessions"):
        new_id = get_session_id()
        session_conversation_data["sessions"].append({"session_id": new_id, "summary": "New Session", "messages": []})
        with open(conversations_file, 'w') as fw:
            json.dump(session_conversation_data, fw, indent=4)

# Initialize Streamlit session state
if 'current_session_id' not in st.session_state:
    first_session = session_conversation_data['sessions'][0]
    st.session_state['current_session_id'] = first_session['session_id']
    st.session_state['summary'] = first_session['summary']
    st.session_state['messages'] = first_session['messages']

team = agents.get_agent_team(st.session_state['current_session_id'])

# Callback to switch sessions
def switch_session():
    selected_summary = st.session_state['sidebar_select']
    selected_id = next(s['session_id'] for s in session_conversation_data['sessions']
                       if s['summary'] == selected_summary)
    st.session_state['current_session_id'] = selected_id
    for sess in session_conversation_data['sessions']:
        if sess['session_id'] == selected_id:
            st.session_state['messages'] = sess.get('messages', [])
            st.session_state['summary'] = sess.get('summary', "No Summary")
            break

# UI
def render_sidebar():
    with st.sidebar:
        st.title("**PodcastAI.**")
        if st.button(":heavy_plus_sign: New Chat"):
            # create and switch to new session
            new_id = get_session_id()
            session_conversation_data['sessions'].insert(0, {"session_id": new_id, "summary": "New Session", "messages": []})
            st.session_state['current_session_id'] = new_id
            st.session_state['messages'] = []
            # persist
            with open(conversations_file, 'w') as f:
                json.dump(session_conversation_data, f, indent=4)

        session_id_map = {s['summary']: s['session_id'] for s in session_conversation_data['sessions']}
        summaries = list(session_id_map.keys())
        current_summary = next(s['summary'] for s in session_conversation_data['sessions']
                            if s['session_id'] == st.session_state['current_session_id'])

        st.selectbox(
            "Sessions",
            summaries,
            index=summaries.index(current_summary),
            key='sidebar_select',
            on_change=switch_session
        )


def restore_session():
    st.title("PodcastAI.")
    for message in st.session_state['messages']:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

# Chat input
def render_body():
    if prompt := st.chat_input("Enter a podcast idea..."):
        # display user
        with st.chat_message("user"):
            st.markdown(prompt)
        # append user message
        user_entry = {"role": "user", "content": prompt}
        st.session_state['messages'].append(user_entry)
        # update persistent data
        for sess in session_conversation_data['sessions']:
            if sess['session_id'] == st.session_state['current_session_id']:
                sess['messages'].append(user_entry)
                break

        # assistant response
        with st.chat_message("assistant"):
            placeholder = st.empty()
            cursor_placeholder = st.empty()

            with cursor_placeholder:
                st_lottie(load_json("assets/lotte-loading.json"), height=60)

            response: Iterator[RunResponse] = team.run(
                prompt,
                stream=True,
                session_id=st.session_state['current_session_id']
            )
            full_text = ""
            for chunk in response:
                if hasattr(chunk, 'content') and chunk.content:
                    full_text += chunk.content
                    placeholder.markdown(strip_think_sections(full_text))
                    time.sleep(0.005)

            cursor_placeholder.empty()

        assistant_entry = {"role": "assistant", "content": strip_think_sections(full_text)}
        st.session_state['messages'].append(assistant_entry)
        for sess in session_conversation_data['sessions']:
            if sess['session_id'] == st.session_state['current_session_id']:
                sess['messages'].append(assistant_entry)
                break

        current_session = next(s for s in session_conversation_data['sessions']
                       if s['session_id'] == st.session_state['current_session_id'])

        if current_session.get("summary", "New Session") == "New Session":
            summary_line = strip_think_sections(full_text).splitlines()[0]
            
            summary_text = ""
            resp: Iterator[RunResponse] = agents.get_summary_agent(session_id="fd").run(summary_line, stream=True)
            for chunk in resp:
                if hasattr(chunk, 'content') and chunk.content:
                    summary_text += chunk.content
        
            current_session["summary"] = summary_text
            st.session_state['summary'] = summary_text

        # persist conversations
        with open(conversations_file, "w") as f:
            json.dump(session_conversation_data, f, indent=4)


def main():
    render_sidebar()
    restore_session()
    render_body()


if __name__ == "__main__":
    main()
