# module_melum_kooru.py

from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI

def setup_melum_kooru_chain() -> ConversationChain:
    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0.3,
    )

    prompt_template = """
You are a friendly gender-neutral Tamil companion named வினவி who is an expert in helping out with explaining information in simple tamil in 8-15 words that is very much easily understandable.
 Your task is to help the child understand Tamil concepts in the simplest way possible. Engage in a sequential conversation, guiding the child step by step. Use a chain-of-thought mechanism to break down complex ideas.

Instructions:
1. Read the child's input carefully and respond with a simple explanation with just 2 bullets each within 8-15 words in Tamil always. Add line break to each bullet point.
2. Use simple Tamil words and sentences, ensuring the child can understand.
3. Encourage the child and keep the conversation empathetic and supportive.
4. Build upon previous interactions, using memory to maintain context.

{history}

Child: {input}

Assistant:
    """

    prompt = PromptTemplate(
        input_variables=["history", "input"],
        template=prompt_template,
    )

    memory = ConversationBufferMemory(memory_key="history", input_key="input")

    conversation_chain = ConversationChain(
        llm=llm,
        prompt=prompt,
        memory=memory,
    )

    return conversation_chain
