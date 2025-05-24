# This is a python comment.

from agno.agent import Agent
from textwrap import dedent
from agno.models.ollama import Ollama
from agno.tools.googlesearch import GoogleSearchTools
from agno.tools.reasoning import ReasoningTools

model_string = "qwen3:8b"
debug_mode = True

def get_content_strategist_agent():
    return Agent(
        model = Ollama(
            id = model_string,
            options = {
                "temperature": 0.7,
                "top_p": 0.9
            }
        ),
        tools = [
            GoogleSearchTools(fixed_max_results=5, fixed_language="en"),
            ReasoningTools(
                add_instructions=True,
                analyze = True,
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
            - Suggest 5 podcast topics.
            - For each topic, make a 2 sentence desciption and justification of the topic in terms of how audience will react. 
                - Use the following format for each:
                    - Description: <description>
                    - Justification: <justification>
        """),
        markdown = True,
        show_tool_calls = True,
        debug_mode = debug_mode
    )

content_strategist_agent = get_content_strategist_agent()
content_strategist_agent.print_response(
    dedent("""\
        Suggest me a podcast topic based on the space industry focusing on the youth who are interested in building
        a career in the space industry. Also the podcast is required to be structured to include fascinating groundbreaking research.
        Make sure that the topics are well thought and social media friendly.
    """),
    stream = True
)
