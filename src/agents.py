# This is a python comment.

from typing import Iterator
from agno.agent import Agent, RunResponse
from textwrap import dedent
from agno.models.ollama import Ollama
from agno.models.google import Gemini

from agno.tools.googlesearch import GoogleSearchTools
from agno.tools.reasoning import ReasoningTools
from agno.tools.arxiv import ArxivTools
from agno.tools.newspaper4k import Newspaper4kTools

from agno.team import Team

# ollama_model_string = "qwen3:8b"
ollama_model_string = "qwen3:1.7b"
gemini_model_string = "gemini-2.0-flash-lite"

ollama_model = Ollama(
    id = ollama_model_string,
    options = {
        "temperature": 0.7,
        "top_p": 0.9
    },
)

gemini_model = Gemini(
    id="gemini-2.0-flash-lite"
)

debug_mode = True

def get_content_strategist_agent():
    return Agent(
        name = "Topic Strategist Agent",
        description = dedent("""\
            You are seasoned content strategist for podcasts with deep and comprehensive expertise
            in strategically analysing the current trend in the market and suggesting a handful of niche topics
            to podcast that are optimized to attract people's attention gaining more clicks.
        """),
        model = gemini_model,
        tools = [
            GoogleSearchTools(fixed_max_results=5, fixed_language="en"),
            ReasoningTools(
                add_instructions = True,
            ),
        ],
        instructions = dedent("""\
            You are seasoned content strategist for podcasts with deep and comprehensive expertise
            in strategically analysing the current trend in the market and suggesting a handful of niche topics
            to podcast that are optimized to attract people's attention gaining more clicks.

            The suggestions are required to follow these criteria:
            - While analysing the trends, use up to date information.
            - The topics should be strategic, creative and considers target audience's interests.
            - The topic should not be overdone.
            - For every topic suggested, justify with a 1 - 2 sentences.

            Your output should follow the following structure:
            - Suggest any number of podcast topics that the prompt tells you to do. If not mentioned, 
            - For each topic, make a 2 sentence desciption and justification of the topic in terms of how audience will react. 
                - Use the following format for each:
                    - Title: <title>
                    - Description: <description>
                    - Justification: <justification>
        """),
        markdown = True,
        show_tool_calls = True,
        debug_mode = debug_mode,
    )

def get_content_writer_agent():
    return Agent(
        name = "Content Writer Agent",
        description = dedent("You are an experienced script writer for a podcast."),
        model = gemini_model,
        tools = [
            ArxivTools(),
            GoogleSearchTools(fixed_language='en'),
            Newspaper4kTools()
        ],
        instructions = dedent("""
            You are an experienced script writer for a podcast. You write podcasts with deep knowledge of
            of the topic specified in the prompt preferrably using tools to fact check the specific topics if required.
            Use Arxiv / Newspaper Tool if required to fact check something or for detailed researched information of the topic.
            
            The script for the podcast is required to be:
            - Completely error free, factual and no false information
            - Contained with miscellaneous things that are required to do in the set during shoot or in post production.
            - Start with a great hook in the beginning and end with a pleasing note such that users would look for more podcasts.
                              
            The script should follow the following general structure:
            - Hook / Short intro
            - Actual intro (longer than first one)
            - Summary of chapters
            - Scripts of all chapters
            - Conclusion / Outro
                              
            Add host narration, sound cue suggestions (e.g., 🎧 “transition music”), and make the tone conversational but informative — like a 
            blend of Radiolab and Lex Fridman.
                              
            If the prompt asks for a different structure, strictly follow that while providing suggestions to make it better 
            at the end.
        """),
        markdown = True,
        debug_mode = True,
        show_tool_calls = True
    )

def get_agent_team():
    return Team(
        name = "PodcastAI Agent Team",
        members = [get_content_strategist_agent(), get_content_writer_agent()],
        mode = "route",
        enable_team_history = True,
        debug_mode=True,
        markdown=True,
        model=gemini_model,
        show_members_responses = True,

        instructions = dedent("""\
            You are the lead Podcast director responsible for classifiying and routing inquiries.
            Carefully analyse each inquiry and determine if it is:
            - a Podcast topic suggestion
            - a Podcast script suggestion
                              
            - For topic suggession inquiries, route to the topic strategist agent
            - For script suggestion inquiries, route to the content writer agent. 

            Always provide a clear explanation of why you're routing the inquiry to a specific agent. 
            Ensure a seamless experience for the user by maintaining context throughout the conversation.
        """)
    )

# resp = get_agent_team().run(
#     dedent("Generate a podcast script with clear, engaging narration that educates listeners about RISC-V. Break complex topics down simply. Include a hook, history, technical highlights, comparisons, and future outlook."),
#     stream = True,
#     stream_intermediate_steps = True,
# )

# print("==================================================== freoahugr;ouahfda;sgfdn;afhuolreafhuo;saijlkfds")
# full_text = ""
# full_textm = ""
# i = 0
# for chunk in resp:
#     print(f"Chunk: {i}")
#     i += 1
#     if hasattr(chunk, 'content') and chunk.content:
#         full_text += chunk.content
#         # full_textm += chunk.messages[-1].content

# print("==================================================== content")
# print(full_text)
# print("==================================================== message")
# print(full_textm)

# print("Response type:", type(resp))
# print("Response content:", resp.content if hasattr(resp, 'content') else str(resp))

