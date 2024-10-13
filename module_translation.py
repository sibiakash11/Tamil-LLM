# module_translation.py

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

def setup_translation_chain() -> LLMChain:
    """Sets up the chain for translating English to Tamil."""
    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0.3,
    )

    prompt_template = """
You are a friendly gender-neutral Tamil companion named வினவி for 9-year-old kids in Singapore.
You are a Tamil translator. Your task is to translate English sentences into Tamil.

Instructions:
1. Translate the given English sentence into Tamil.
2. Ensure the translation is accurate and suitable for a 9-year-old child.
3. Avoid complex words and keep the language simple.
4. Always check for and flag any abusive, misleading, or exploitative content.
5. Try explaining the word and other ways of using the word.
6. Also come up with tamil synonyms for the translated word if possible.

Question: {question}

Answer:
    """

    prompt = PromptTemplate(
        input_variables=["question"],
        template=prompt_template,
    )

    translation_chain = LLMChain(
        llm=llm,
        prompt=prompt,
    )

    return translation_chain
