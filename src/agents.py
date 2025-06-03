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
from agno.tools.cartesia import CartesiaTools
from agno.utils.audio import write_audio_to_file

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
    id = "meta-llama/llama-4-scout-17b-16-16e-instruct",
    temperature = 0.7,
    top_p=0.9
)

content_strategist_model = gemini_model
content_writer_model = gemini_model
content_caption_writer_model = gemini_model
voice_agent_model = gemini_model

summary_agent = gemini_model

debug_mode = True

storage_db_file = "temp/shared_storage.db"
shared_storage = SqliteStorage(table_name="shared_storage", db_file=storage_db_file)

team_memory = Memory(db=shared_storage)

def get_summary_agent(session_id: str):
    return Agent(
        model=gemini_model,
        instructions=dedent("""\
            Your job is to summarize given text in 5 words or less.
            The format of the response should not contain anything but the summary.
        """)
    )

def get_content_strategist_agent(session_id: str):
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
        memory=team_memory,

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
                    # Topic {number}: <title>
                    - **Description**: <description>
                    - **Justification**: <justification>
            
            IMPORTANT: Always number your topics clearly (1, 2, 3, etc.) so they can be referenced later.
            IMPORTANT: Just reply the topics requested, do not include the chain of thoughts in the response
        """),
        markdown = True,
        show_tool_calls = True,
        debug_mode = debug_mode,
    )

def get_content_writer_agent(session_id: str):
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
        memory=team_memory,

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
            IMPORTANT: Just reply the topics requested, do not include the chain of thoughts in the response
        """),
        markdown = True,
        debug_mode = True,
        show_tool_calls = True
    )


def get_content_caption_writer_agent(session_id: str):
    return Agent(
        name = "Caption Writer Agent",
        description = dedent("You are an experienced caption writer advertising podcasts in social media."),
        model = content_caption_writer_model,

        storage=shared_storage,
        add_history_to_messages=True,
        memory=team_memory,

        instructions = dedent("""
            You are an experienced caption writer advertising podcasts in social media.
            IMPORTANT: When asked to write a caption for a numbered topic (like "3rd topic"), first check the conversation
            history to see what topics were previously suggested, then write the caption for the specified topic.
            
            The caption for the podcast is required to be:
            - Completely error free, doesn't spoil the actual podcast script but still gives a "hint" to it.
            - Catchy, should be visually appealing for social media usecases. Use emojis to make it more lively
            - Start with a great hook in the beginning to ensure high click rate
            - Should have a perfect length to be posted as a caption for a post in instagram, twitter or youtube.
            - Make sure that the caption is not cluttered, use indentation if possible
            
            IMPORTANT: Just reply the topics requested, do not include the chain of thoughts in the response
        """),
        markdown = True,
        debug_mode = True,
        show_tool_calls = True
    )

def get_voice_agent(session_id: str):
    return Agent(
        name = "Text to Speech agent",
        description = dedent("Text to speech"),
        model = voice_agent_model,
        tools = [CartesiaTools()],

        storage=shared_storage,
        add_history_to_messages=True,
        memory=team_memory,

        instructions = dedent("""
            You are an Text-To-Speech agent that is reponsible to generating audio files from a given podcast script.

            Strip out the unnessary parts of the script that are usually not required to be spoken like the titles, index or table of contents etc. 
            from the text that are to be converted to speech                  
                              
            IMPORTANT: Make sure to generate only one audio file containing the entire script.
            
        """),
        debug_mode = True,
        show_tool_calls = True
    )

def get_agent_team(session_id: str):
    return Team(
        name = "PodcastAI Agent Team",
        members = [
            get_content_strategist_agent(session_id), 
            get_content_writer_agent(session_id), 
            get_content_caption_writer_agent(session_id),
            get_voice_agent(session_id)
        ],
        mode = "route",
        debug_mode=True,
        markdown=True,
        model=gemini_model,
        show_members_responses = True,

        memory=team_memory,
        storage=shared_storage,
        enable_team_history = True,
        num_of_interactions_from_history=10,  # Increased to capture more context

        session_id=session_id,
        session_state={"session_id": session_id},

        instructions = dedent("""\
            You are the lead Podcast director responsible for classifiying and routing inquiries.
            Carefully analyse each inquiry and determine if it is:
            - a Podcast topic suggestion
            - a Podcast script suggestion
            - a Podcast caption suggestion
            - a Podcast audio suggesion
                              
            - For topic suggession inquiries, route to the topic strategist agent
            - For script suggestion inquiries, route to the content writer agent. 
            - For caption suggession inquiries, route to the caption writer agent
            - For audio suggestion inquiries, route to the text to speech agent. 
                              
            IMPORTANT: When routing to the text to speech agent, forward the RAW podcast script text from history
            to the text to speech agent as its expexted output so that it gets context to create audio files. 

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
                              
            IMPORTANT: Just reply the topics requested, do not include the chain of thoughts in the response and only forward the 
            response of the member agents and do not include your own response.
        """)
    )

if __name__ == "__main__":

    import dotenv
    dotenv.load_dotenv()
    # team = get_agent_team()
    
    # print("=== First Request: Getting Topics ===")
    # team.print_response("Suggest 5 podcast topics about tech trends.", stream=True, stream_intermediate_steps=True)
    
    # print("\n" + "=" * 100)
    # print("=== Storage Sessions ===")
    # print(shared_storage.get_all_sessions())
    
    # print("\n" + "=" * 100)
    # print("=== Second Request: Script for 3rd Topic ===")
    # team.print_response("Write a podcast script for the 3rd topic you mentioned.", stream=True, stream_intermediate_steps=True)


    # print(car.list_voices())

    # sample = dedent("""
    #     Convert the following in to audio. Use different voices for different characters. Return one audio file containing the whole script.
        
    #     \"HHey everyone, welcome back to Mind Minutes. I'm Sam, and today we're diving into a question we've all asked ourselves: Why do I keep procrastinating?
    #     Here's the truth — procrastination isn't laziness. It's often fear. Fear of failure, of imperfection, or even of success. Crazy, right?
    #     Your brain seeks short-term comfort over long-term progress. That's why scrolling feels easier than starting that big project.
    #     So here's one tip that works: set a 5-minute timer. Tell yourself you only have to start. Once you do, momentum kicks in.
    #     That's it for today. Tiny steps, big change.
    #     Catch you in the next episode of Mind Minutes.\"
    # """)

    # response = get_voice_agent("fdsafds").run(sample)
    # if response.audio:
    #     for i in range(len(response.audio)):
    #         write_audio_to_file(
    #             response.audio[i].base64_audio,
    #             filename=f"sample_sample{i}.mp3",
    #         )