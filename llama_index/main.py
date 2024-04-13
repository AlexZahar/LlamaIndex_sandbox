import os
value = os.getenv('OPENAI_API_KEY')

from llama_index.llms.openai import OpenAI
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    load_index_from_storage,
    StorageContext,
    ServiceContext,
    set_global_service_context,
    Settings
)

llm = OpenAI(model="gpt-3.5-turbo", temperature=1, max_tokens=356)

documents = SimpleDirectoryReader("books").load_data()

service_context = ServiceContext.from_defaults(llm=llm, chunk_size=1000, chunk_overlap=20)

index = VectorStoreIndex.from_documents(documents, service_context=service_context)

# Persist the index to disk
index.storage_context.persist()

# Load the index from disk
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context=storage_context)


query_engine = index.as_query_engine()  # For memory storage as chatbot use "as_chat_engine"

# response = query_engine.query("Who is the author of Alice In wonderland? Provide more details")
response = query_engine.query("I thought the original book 'Allice in Wonderland' has been written by a different author? Who is the original author of Alice In wonderland?")
# response = query_engine.query("Who is Alexandru Zahar")
# response = query_engine.query("What is the latest version of node")

print(response)
