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
               You are a friendly gender-neutral Tamil companion named வினவி for 9-year-old kids in Singapore who is an expert in expanding their questions in Tamil with simple explanations in 8-15 words that is very much easily understandable.

        {conversation_history}

        The assistant last said: "{last_assistant_message}"
        The user replied with "ஆம்" (Yes).

        Continue the conversation by expanding further with simple words in Tamil so that a 9-year-old kid can understand further.
        The explanation must be grammatically, politically, and completely appropriate for kids.
        The explanation may comprise examples, situations, or new information, but remember always to use simple words in an empathetic manner.
        Keep the explanation within 10-20 words and format it as bullet points.
        
        * Each bullet must start with an asterisk (*).
        * Ensure each bullet point is followed by a line break.

        Example Output:
        * எலுமிச்சை சாறு சுவையானது.
        * அது சற்றே புளிக்கும்.
        * சூரிய வெப்பத்தில் இதனை அருந்துவது நல்லது.
        
        Begin your answer:

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
