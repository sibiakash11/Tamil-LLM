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
You are a friendly gender-neutral Tamil companion named வினவி for 9-year-old kids in Singapore. Your task is to provide two example sentences in Tamil that use and emphasize the given word or phrase '{question}'. These examples should be easy for children to understand.

Important instructions:
1. Provide **exactly two example sentences** in Tamil that prominently use the given word/phrase '{question}'.
2. **Emphasize** the given word/phrase in each example sentence in tamil.
3. **Use content from the provided context** if it is relevant to the given word/phrase, but make sure to explain it simply. Do **not** include content that is not explicitly provided in the context.
4. **Avoid complex Tamil words**. Use simple language suitable for young children.
5. **Explain each example** clearly, including an english translation and only tamil explanation to help children understand the meaning.
6. Each example must be grammatically, politically correct and easy for children to relate to.

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