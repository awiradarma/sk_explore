# Copyright (c) Microsoft. All rights reserved.

import asyncio
from functools import reduce
from typing import Annotated

from openai import AsyncOpenAI
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_completion import OpenAIChatCompletion
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.kernel import Kernel
from semantic_kernel.functions.kernel_function_decorator import kernel_function

###################################################################
# The following sample demonstrates how to create a simple,       #
# non-group agent that repeats the user message in the voice      #
# of a pirate and then ends with a parrot sound.                  #
###################################################################

# To toggle streaming or non-streaming mode, change the following boolean
streaming = False

# Define the agent name and instructions
AGENT_NAME = "Host"
AGENT_INSTRUCTIONS = "Answer questions about the menu."

# Define a sample plugin for the sample
class MenuPlugin:
    """A sample Menu Plugin used for the concept sample."""

    @kernel_function(description="Provides a list of specials from the menu.")
    def get_specials(self) -> Annotated[str, "Returns the specials from the menu."]:
        return """
        Special Soup: Clam Chowder
        Special Salad: Cobb Salad
        Special Drink: Chai Tea
        """

    @kernel_function(description="Provides the price of the requested menu item.")
    def get_item_price(
        self, menu_item: Annotated[str, "The name of the menu item."]
    ) -> Annotated[str, "Returns the price of the menu item."]:
        return "$9.99"
    
async def invoke_agent(agent: ChatCompletionAgent, input: str, chat: ChatHistory):
    """Invoke the agent with the user input."""
    chat.add_user_message(input)

    print(f"# {AuthorRole.USER}: '{input}'")

    if streaming:
        contents = []
        content_name = ""
        async for content in agent.invoke_stream(chat):
            content_name = content.name
            contents.append(content)
        streaming_chat_message = reduce(lambda first, second: first + second, contents)
        print(f"# {content.role} - {content_name or '*'}: '{streaming_chat_message}'")
        chat.add_message(streaming_chat_message)
    else:
        async for content in agent.invoke(chat):
            print(f"# {content.role} - {content.name or '*'}: '{content.content}'")
            chat.add_message(content)


async def main():
    # Create the instance of the Kernel
    kernel = Kernel()

    openAIClient: AsyncOpenAI = AsyncOpenAI(
    api_key="fake-key",  # This cannot be an empty string, use a fake key
    base_url="http://localhost:11434/v1",
    # base_url="http://localhost:8080/v1", # llamafile
)

    # Add the OpenAIChatCompletion AI Service to the Kernel
    kernel.add_service(OpenAIChatCompletion(service_id="agent", ai_model_id="llama3.1", async_client=openAIClient))
    settings = kernel.get_prompt_execution_settings_from_service_id(service_id="agent")
    settings.max_tokens = 2000
    settings.temperature = 0.7
    settings.top_p = 0.8
    settings.function_choice_behavior = FunctionChoiceBehavior.Auto()
    kernel.add_plugin(MenuPlugin(), plugin_name="menu")

    # Create the agent
    agent = ChatCompletionAgent(service_id="agent", kernel=kernel, name=AGENT_NAME, instructions=AGENT_INSTRUCTIONS, execution_settings=settings)

    # Define the chat history
    chat = ChatHistory()

    # Respond to user input
    # await invoke_agent(agent, "Hello", chat)
    await invoke_agent(agent, "What are your specials?", chat)
    await invoke_agent(agent, "How much is Clam Chowder soup?", chat)
    await invoke_agent(agent, "How much is the t-bone steak?", chat)
    await invoke_agent(agent, "What is the special drink?", chat)
    await invoke_agent(agent, "Thank you", chat)

if __name__ == "__main__":
    asyncio.run(main())