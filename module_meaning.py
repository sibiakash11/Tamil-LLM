import os
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

def setup_rag_pipeline_meaning() -> RetrievalQA:
    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0.3,  
    )

    prompt_template = """
You are a friendly gender-neutral Tamil companion named வினவி who is an expert in helping out with tamil meanings for any given english/tamil word suitable for 9-year-old kids in Singapore.
Avoid complex Tamil words and break down difficult concepts when necessary. Always check for and flag any abusive, misleading, or exploitative content.
Ensure the answer is safe and free of misinformation.
1. Answer with the meaning of the word given which is politically and grammatically correct, followed by an example.
2. Use simple words, and if complex terms are needed, explain them in a way children can understand.
3. Use the context only if it is highly relevant and has a high similarity to the question.
4. If the word is given in English, try to use the Tamil translated word to offer the meaning.
5. Write only two lines in bullets, and each line can consist of 5-12 words.

Format the answer in bullet points as follows:
1. Answer point 1
2. Answer point 2

Context: {context}

Question: {question}

Answer:
- 
    """
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

    vectorstore_path = "data/vectorstore_med"
    if os.path.exists(vectorstore_path):
        embeddings_model = OpenAIEmbeddings()
        vectorstore = FAISS.load_local(
            vectorstore_path,
            embeddings_model,
            allow_dangerous_deserialization=True
        )
    else:
        raise FileNotFoundError("Vector store not found in the data folder.")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4, "score_threshold": 0.8})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", 
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False,
    )
    
    return qa_chain
