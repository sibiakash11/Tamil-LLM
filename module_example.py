import os
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA  # Using RetrievalQA for RAG pipeline
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

def setup_rag_pipeline_example() -> RetrievalQA:
    """Sets up the RAG pipeline for 'Provide an example in Tamil within Singapore context'."""
    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0.3,
        max_tokens=250,  # Increased token limit for a more detailed response
    )

    prompt_template = """
You are a friendly gender-neutral Tamil companion named வினவி who is an expert in helping out with tamil examples for any given english/tamil word suitable for 9-year-old kids in Singapore. You will strictly answer every question in tamil language. 
English words should only be used in the translation, other than that the whole response should be completely in simple tamil.
Your task is to provide an example sentences in Tamil that use the given word or phrase taken from the '{question}' within 7-12 words altogether. These examples should be easy for children to understand.

Important instructions:
1. Provide exactly one sentence in Tamil that prominently uses the given word/phrase '{question}'.
2. Use the given word/phrase in the example sentence in Tamil without bold or special formatting.
3. **Use content from the provided context** if it is relevant to the given word/phrase, but make sure to explain it simply. Do **not** include content that is not explicitly provided in the context.
4. **Avoid complex Tamil words**. Use simple language suitable for young children.
5. **Explain the example** clearly in only Tamil in simple words in a way that a kid would understand starting with "விளக்கம்:".
6. Each example must be grammatically, politically correct and easy for children to relate to.
7. Finally, give an English translation of the same example sentence starting with "
ஆங்கில மொழிபெயர்ப்பு:" .
Just give the direct example with a line of explanation without any titles as an answer.

Context: {context}

Answer:
"""

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

    # Load the vector store
    vectorstore_path = "data/vectorstore_med"
    if os.path.exists(vectorstore_path):
        embeddings_model = OpenAIEmbeddings()
        vectorstore = FAISS.load_local(
            vectorstore_path,
            embeddings_model,
            allow_dangerous_deserialization=True  # Adjusted for safety
        )
    else:
        raise FileNotFoundError("Vector store 'vectorstore_med' not found in the data folder.")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # 'stuff' chain type concatenates retrieved docs
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False,
    )
    return qa_chain