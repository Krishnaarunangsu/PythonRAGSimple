from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
from langchain_core.prompts import ChatPromptTemplate

# Load model on CPU - NO DTYPE PARAMETERS
pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device=-1,  # CPU
    max_new_tokens=200
)

llm = HuggingFacePipeline(pipeline=pipe)

# Create simple chain
prompt = ChatPromptTemplate.from_template("Question: {q}\nAnswer: ")
chain = prompt | llm

# Ask a question
result = chain.invoke({"q": "What is machine learning?"})
print(result)