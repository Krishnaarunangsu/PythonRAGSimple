# Core LangChain imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

# Hugging Face specific imports for CPU
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Document processing imports
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# ============================================
# CPU-OPTIMIZED MODELS (NO DTYPE PARAMETERS)
# ============================================

# Option 1: TinyLlama (1.1B) - Best for CPU
def load_tinyllama_cpu():
    """Load TinyLlama model optimized for CPU - no dtype parameters"""
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    # Simple loading without any dtype parameters
    hf_llm = HuggingFacePipeline.from_model_id(
        model_id=model_id,
        task="text-generation",
        device=-1,  # -1 means CPU
        pipeline_kwargs={
            "max_new_tokens": 256,
            "temperature": 0.7,
            "do_sample": True,
            "top_p": 0.95,
            "top_k": 50,
            "repetition_penalty": 1.1,
        }
        # No model_kwargs with dtype!
    )
    return hf_llm


# Option 2: Even Simpler - Using pipeline directly
def load_tinyllama_simple():
    """
    Even simpler loading with pipeline
    Returns:

    """
    pipe = pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device=-1,  # CPU
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True,
        top_p=0.95,
    )
    return HuggingFacePipeline(pipeline=pipe)


# Option 3: GPT-2 (124M) - Fastest option
def load_gpt2_cpu():
    """Load GPT-2 - fastest for CPU"""
    pipe = pipeline(
        "text-generation",
        model="gpt2",
        device=-1,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True,
    )
    return HuggingFacePipeline(pipeline=pipe)


# Option 4: Phi-2 (2.7B) - Better quality
def load_phi2_cpu():
    """Load Microsoft Phi-2 - better quality, slower"""
    model_id = "microsoft/phi-2"

    # Load with trust_remote_code=True but NO dtype
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        low_cpu_mem_usage=True
        # No dtype parameter!
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True,
    )

    return HuggingFacePipeline(pipeline=pipe)


# Option 5: DistilGPT2 - Very lightweight
def load_distilgpt2_cpu():
    """Load DistilGPT2 - extremely lightweight"""
    pipe = pipeline(
        "text-generation",
        model="distilgpt2",
        device=-1,
        max_new_tokens=150,
        temperature=0.7,
    )
    return HuggingFacePipeline(pipeline=pipe)


# ============================================
# CPU EMBEDDINGS (NO GPU REQUIRED)
# ============================================

def get_cpu_embeddings():
    """Lightweight embeddings model for CPU"""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",  # Small, fast
        model_kwargs={'device': 'cpu'},  # Force CPU
    )


# ============================================
# COMPLETE RAG PIPELINE FOR CPU
# ============================================

def create_cpu_rag_pipeline():
    """Create a complete RAG pipeline optimized for CPU"""

    # 1. Load a CPU-optimized model (choose one)
    print("Loading TinyLlama model on CPU...")
    llm = load_tinyllama_simple()  # Using the simplest loading method

    # 2. Create prompt template
    prompt = ChatPromptTemplate.from_template("""
    You are a helpful AI assistant. Answer the question based on the provided context.
    If the answer cannot be found in the context, say "I don't have enough information to answer that."

    Context: {context}

    Question: {input}

    Answer:
    """)

    # 3. Create sample documents (for testing)
    sample_texts = [
        "Artificial intelligence (AI) is the simulation of human intelligence in machines that are programmed to think and learn.",
        "Machine learning is a subset of AI that enables systems to automatically learn and improve from experience without explicit programming.",
        "Neural networks are computing systems inspired by biological neural networks that constitute animal brains.",
        "Deep learning uses multiple layers in neural networks to progressively extract higher-level features from raw input.",
        "Natural language processing (NLP) helps computers understand, interpret and manipulate human language."
    ]

    # 4. Create embeddings and vector store on CPU
    print("Creating embeddings on CPU...")
    embeddings = get_cpu_embeddings()

    # Convert to documents
    from langchain_core.documents import Document
    documents = [Document(page_content=text) for text in sample_texts]

    # Create vector store
    print("Building vector store...")
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # 5. Create chains
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

    return retrieval_chain, llm


# ============================================
# MAIN APPLICATION
# ============================================

def main():
    print("=" * 60)
    print("CPU-ONLY LOCAL LLM WITH RAG")
    print("=" * 60)
    print("No GPU required - runs entirely on CPU")
    print("=" * 60)

    # Create the RAG pipeline
    print("\n[1/3] Setting up RAG pipeline...")
    rag_chain, llm = create_cpu_rag_pipeline()

    print("\n[2/3] Ready! You can now ask questions.")
    print("[3/3] Type 'quit' to exit.\n")

    # Interactive Q&A loop
    while True:
        # Get user input
        question = input("\n📝 Your question: ").strip()

        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye! 👋")
            break

        if not question:
            continue

        print("\n🤔 Thinking...")

        try:
            # Get response from RAG chain
            response = rag_chain.invoke({
                "input": question
            })

            # Print answer
            print("\n💡 Answer:", response['answer'])

            # Optional: Show sources
            if 'context' in response and response['context']:
                print("\n📚 Sources used:")
                for i, doc in enumerate(response['context'], 1):
                    preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                    print(f"   {i}. {preview}")

        except Exception as e:
            print(f"\n❌ Error: {e}")


# ============================================
# SIMPLE TEST FUNCTION
# ============================================

def quick_test():
    """Minimal test to verify everything works"""
    print("Running quick test...")

    # Simplest possible setup
    pipe = pipeline(
        "text-generation",
        model="gpt2",  # Smallest model
        device=-1,
        max_new_tokens=50
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    # Simple prompt
    prompt = ChatPromptTemplate.from_template("Q: {question}\nA: ")
    chain = prompt | llm

    # Test
    result = chain.invoke({"question": "What is AI?"})
    print(f"Test result: {result}")
    print("Quick test passed!")


# ============================================
# LOAD FROM LOCAL PATH (NO DTYPE)
# ============================================

def load_local_model(model_path):
    """Load a model from a local path - no dtype parameters"""

    print(f"Loading model from: {model_path}")

    # Load without any dtype specification
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True
        # No dtype parameter!
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.7,
    )

    return HuggingFacePipeline(pipeline=pipe)


# ============================================
# BATCH PROCESSING EXAMPLE
# ============================================

def batch_processing_example():
    """Process multiple questions at once"""

    print("\n" + "=" * 50)
    print("Batch Processing Example")
    print("=" * 50)

    # Load model
    llm = load_distilgpt2_cpu()  # Using smallest model for speed

    # Multiple questions
    questions = [
        "What is Python?",
        "What is JavaScript?",
        "What is HTML?"
    ]

    print(f"Processing {len(questions)} questions...")

    # Process in batch
    responses = llm.batch(questions)

    for q, r in zip(questions, responses):
        print(f"\nQ: {q}")
        print(f"A: {r[:100]}...")  # First 100 chars


# ============================================
# STREAMING EXAMPLE
# ============================================

def streaming_example():
    """Example of streaming responses"""

    print("\n" + "=" * 50)
    print("Streaming Response Example")
    print("=" * 50)

    # Load model
    llm = load_tinyllama_simple()

    prompt = "Write a short poem about coding:"
    print(f"Prompt: {prompt}\n")
    print("Response: ", end="", flush=True)

    # Stream the response
    for chunk in llm.stream(prompt):
        print(chunk, end="", flush=True)

    print("\n" + "=" * 50)


# ============================================
# RUN THE APPLICATION
# ============================================

if __name__ == "__main__":
    import sys

    # Choose what to run
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            quick_test()
        elif sys.argv[1] == "--batch":
            batch_processing_example()
        elif sys.argv[1] == "--stream":
            streaming_example()
        else:
            print("Usage: python script.py [--test|--batch|--stream]")
    else:
        # Default: Run interactive main app
        main()