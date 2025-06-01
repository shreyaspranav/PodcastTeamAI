# This is a python comment.

from typing import Iterator
from agno.agent import Agent, RunResponse
from textwrap import dedent

from agno.models.ollama import Ollama
from agno.models.google import Gemini
from agno.models.groq import Groq

from agno.tools.googlesearch import GoogleSearchTools
from agno.tools.reasoning import ReasoningTools
from agno.tools.arxiv import ArxivTools
from agno.tools.newspaper4k import Newspaper4kTools

from agno.memory.v2.memory import Memory
from agno.storage.sqlite import SqliteStorage

from agno.team import Team

ollama_model = Ollama(
    id = "qwen3:1.7b",
    options = {
        "temperature": 0.7,
        "top_p": 0.9
    },
)

gemini_model = Gemini(
    id="gemini-2.0-flash-lite",
    temperature=0.7,
    top_k=70
)

groq_model = Groq(
    id = "meta-llama/llama-4-scout-17b-16e-instruct",
    temperature = 0.7,
    top_p=0.9
)

content_strategist_model = gemini_model
content_writer_model = groq_model

debug_mode = True

storage_db_file = "temp/shared_storage.db"
shared_storage = SqliteStorage(table_name="shared_storage", db_file=storage_db_file)

team_memory = Memory(db=shared_storage)

def get_content_strategist_agent():
    return Agent(
        name = "Topic Strategist Agent",
        description = dedent("""\
            You are seasoned content strategist for podcasts with deep and comprehensive expertise
            in strategically analysing the current trend in the market and suggesting a handful of niche topics
            to podcast that are optimized to attract people's attention gaining more clicks.
        """),
        model = content_strategist_model,
        tools = [
            GoogleSearchTools(fixed_max_results=5, fixed_language="en"),
            ReasoningTools(
                add_instructions = True,
            ),
        ],
        storage=shared_storage,
        add_history_to_messages=True,
        instructions = dedent("""\
            You are seasoned content strategist for podcasts with deep and comprehensive expertise
            in strategically analysing the current trend in the market and suggesting a handful of niche topics
            to podcast that are optimized to attract people's attention gaining more clicks.

            The suggestions are required to follow these criteria:
            - While analysing the trends, use up to date information.
            - The topics should be strategic, creative and considers target audience's interests.
            - The topic should not be overdone.
            - For every topic suggested, justify with a 1 - 2 sentences. The justification should be convicing to the user.

            Your output should follow the following structure:
            - Suggest any number of podcast topics that the prompt tells you to do. If not mentioned, suggest 5.
            - For each topic, make a 2 sentence desciption and justification of the topic in terms of how audience will react. 
                - Use the following format for each:
                    ## Topic {number}: <title>
                    - **Description**: <description>
                    - **Justification**: <justification>
            
            IMPORTANT: Always number your topics clearly (1, 2, 3, etc.) so they can be referenced later.
        """),
        markdown = True,
        show_tool_calls = True,
        debug_mode = debug_mode,
    )

def get_content_writer_agent():
    return Agent(
        name = "Content Writer Agent",
        description = dedent("You are an experienced script writer for a podcast."),
        model = content_writer_model,
        tools = [
            ArxivTools(),
            GoogleSearchTools(fixed_language='en'),
            Newspaper4kTools()
        ],

        storage=shared_storage,
        add_history_to_messages=True,

        instructions = dedent("""
            You are an experienced script writer for a podcast. You write podcasts with deep knowledge of
            of the topic specified in the prompt preferrably using tools to fact check the specific topics if required.
            Use Arxiv / Newspaper Tool if required to fact check something or for detailed researched information of the topic.
            
            IMPORTANT: When asked to write a script for a numbered topic (like "3rd topic"), first check the conversation
            history to see what topics were previously suggested, then write the script for the specified topic.
            
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
                              
            Add host narration, sound cue suggestions (e.g., 🎧 "transition music"), and make the tone conversational but informative — like a 
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
        debug_mode=True,
        markdown=True,
        model=gemini_model,
        show_members_responses = True,

        memory=team_memory,
        storage=shared_storage,
        enable_team_history = True,
        num_of_interactions_from_history=10,  # Increased to capture more context

        instructions = dedent("""\
            You are the lead Podcast director responsible for classifiying and routing inquiries.
            Carefully analyse each inquiry and determine if it is:
            - a Podcast topic suggestion
            - a Podcast script suggestion
                              
            - For topic suggession inquiries, route to the topic strategist agent
            - For script suggestion inquiries, route to the content writer agent. 

            IMPORTANT CONTEXT HANDLING:
            - When users reference numbered topics (like "3rd topic", "topic 2", etc.), this means they're referring 
              to topics that were previously suggested in this conversation.
            - Before routing to content writer, include a summary of the conversation context so the writer knows
              which specific topic to write about.
            - If you can't find the referenced topic in history, ask the user to clarify or re-list the topics.

            ROUTING LOGIC:
            1. Check if the request mentions a specific numbered topic
            2. If yes, look for previous topic suggestions in the conversation
            3. Route to content writer with explicit topic information
            4. If no numbered reference, route based on general intent

            Always provide a clear explanation of why you're routing the inquiry to a specific agent. 
            Ensure a seamless experience for the user by maintaining context throughout the conversation.
        """)
    )

if __name__ == "__main__":
    team = get_agent_team()
    
    print("=== First Request: Getting Topics ===")
    team.print_response("Suggest 5 podcast topics about tech trends.", stream=True, stream_intermediate_steps=True)
    
    print("\n" + "=" * 100)
    print("=== Storage Sessions ===")
    print(shared_storage.get_all_sessions())
    
    print("\n" + "=" * 100)
    print("=== Second Request: Script for 3rd Topic ===")
    team.print_response("Write a podcast script for the 3rd topic you mentioned.", stream=True, stream_intermediate_steps=True)