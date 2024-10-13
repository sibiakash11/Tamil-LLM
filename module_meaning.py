import os
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

def setup_rag_pipeline_meaning() -> RetrievalQA:
    """Sets up the RAG pipeline for 'Give a detailed explanation in Tamil' with a high similarity threshold for context relevance."""
    
    # Set up the LLM (GPT-4o)
    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0.3,  # Slight temperature randomness for creativity but remains focused
    )

    # Define the prompt template
    prompt_template = """
You are a friendly gender-neutral Tamil companion named வினவி for 9-year-old kids in Singapore.
Avoid complex Tamil words and break down difficult concepts when necessary. Always check for and flag any abusive, misleading, or exploitative content.
Ensure the answer is safe and free of misinformation.
1. Answer with the meaning of the word given which is policially and grammatically correct, and give an example using the word that is understandable for tamil kids.
2. Use simple words, and if complex terms are needed, explain them in a way children can understand.
3. Use the context only if it is highly relevant and has a high similarity to the question.
4. If the word is given in english try to use the tamil translated word to offer the meaning.

Context: {context}

Question: {question}

Answer:
    """

    # Create the prompt template
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

    # Load the FAISS vector store
    vectorstore_path = "data/vectorstore_med"
    if os.path.exists(vectorstore_path):
        # Load embeddings and vector store
        embeddings_model = OpenAIEmbeddings()
        vectorstore = FAISS.load_local(
            vectorstore_path,
            embeddings_model,
            allow_dangerous_deserialization=True
        )
    else:
        raise FileNotFoundError("Vector store not found in the data folder.")

    # Set up the retriever with a high similarity threshold
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4, "score_threshold": 0.8})

    # Build the RetrievalQA chain with the prompt
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # "Stuff" chain type to concatenate all retrieved documents
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False,  # Don't return the actual documents, just the generated answer
    )
    
    return qa_chain
