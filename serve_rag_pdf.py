#!/usr/bin/env python
from fastapi import FastAPI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langserve import add_routes

import os, json

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "pr-flowery-realization-1"

# Function to load configuration from the JSON file based on model name
def load_config(model_name):
    with open('config.json', 'r') as file:
        configs = json.load(file)
        
    return configs.get(model_name, configs.get('default'))

# Factory function to get the model based on the class name from the config
def get_model(model_class_name, model):
    # Mapping model class names to the actual class
    model_classes = {
        "ChatOpenAI": ChatOpenAI,
        "ChatOllama": ChatOllama,
        # Add more models here as you add them to your config
    }
    
    # Return the corresponding class or raise an error if not found
    if model_class_name in model_classes:
        return model_classes[model_class_name](model=model)
    else:
        raise ValueError(f"Model class '{model_class_name}' not recognized.")

file_path = "example_data/cv_philippos_barabas.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = InMemoryVectorStore.from_documents(
    documents=splits, embedding=OpenAIEmbeddings()
)

retriever = vectorstore.as_retriever()

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)


if __name__ == "__main__":
    import uvicorn

    # Input model name from user
    model_name = input("Enter the model name (ollama, openai): ")
    
    # Load the appropriate configuration
    config = load_config(model_name)

    print(f"Running with config: {config}")

    # Get the model class name from the configuration
    model_class_name = config['model_class']

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{question}"),
        ]
    )

    # 2. Create model
    model = get_model(model_class_name, config['model'])

    # 3. Create parser
    parser = StrOutputParser()

    # 4. Create chain
    #question_answer_chain = create_stuff_documents_chain(model, prompt)
    #rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    # 4. App definition
    app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API server using LangChain's Runnable interfaces",
    )

    # 5. Adding chain route
    add_routes(
        app,
        rag_chain,
        path="/chain",
    )

    uvicorn.run(app, host="localhost", port=8000)
