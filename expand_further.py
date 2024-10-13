# expand_further.py

import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

def setup_expand_further_chain():
    """
    Setup a chain of thought-based pipeline that uses memory of the last 5 conversations
    and expands further on the current context.
    """
    # Define the prompt for expanding the conversation further with chain of thought reasoning
    expand_prompt = PromptTemplate(
        input_variables=["conversation_history", "last_assistant_message"],
        template="""
        You are a friendly gender-neutral Tamil companion named வினவி for 9-year-old kids in Singapore who is an expert in expanding their questions in tamil with simple explanations that is very much easily understandable.

        {conversation_history}

        The assistant last said: "{last_assistant_message}"
        The user replied with "ஆம்" (Yes).

        Continue the conversation by expanding further with simple words in tamil so that a 9 year old kid can understand further.
        The explanation must be grammatically, politically and completely appropriate to kids.
        The explanation may comprise of examples, situations, new information but remember always use simple words in an empathetic manner.
        Keep the explanation within 120 words.

        """
    )

    # Define the chain using ChatGPT model
    expand_chain = LLMChain(
        llm=ChatOpenAI(
            model_name="gpt-4o",
            temperature=0.3,
             max_tokens=200
            
        ),
        prompt=expand_prompt
    )

    return expand_chain
