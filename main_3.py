# Core LangChain imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

# Hugging Face specific imports for CPU
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Document processing imports
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# ============================================
# CPU-OPTIMIZED SMALL MODELS (UPDATED)
# ============================================

# Option 1: TinyLlama (1.1B) - Best for CPU, runs relatively fast
def load_tinyllama_cpu():
    """Load TinyLlama model optimized for CPU"""
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    # Using model_kwargs with 'dtype' instead of 'torch_dtype'
    hf_llm = HuggingFacePipeline.from_model_id(
        model_id=model_id,
        task="text-generation",
        device=-1,  # Force CPU
        pipeline_kwargs={
            "max_new_tokens": 256,  # Lower for faster CPU inference
            "temperature": 0.7,
            "do_sample": True,
            "top_p": 0.95,
            "top_k": 50,
            "repetition_penalty": 1.1,
        },
        model_kwargs={
            "dtype": torch.float32,  # Changed from torch_dtype to dtype
            "low_cpu_mem_usage": True,  # Optimize memory usage
        }
    )
    return hf_llm


# Option 2: Phi-2 (2.7B) - Good quality, needs more RAM
def load_phi2_cpu():
    """Load Microsoft Phi-2 optimized for CPU"""
    model_id = "microsoft/phi-2"

    # Direct model loading with updated parameter
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        dtype=torch.float32,  # Changed from torch_dtype to dtype
        low_cpu_mem_usage=True
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True,
    )

    hf_llm = HuggingFacePipeline(pipeline=pipe)
    return hf_llm


# Option 3: GPT-2 (124M) - Very fast, but lower quality
def load_gpt2_cpu():
    """Load GPT-2 (smallest, fastest)"""
    # GPT-2 is simpler, might not need dtype specification
    hf_llm = HuggingFacePipeline.from_model_id(
        model_id="gpt2",
        task="text-generation",
        device=-1,
        pipeline_kwargs={
            "max_new_tokens": 200,
            "temperature": 0.7,
            "do_sample": True,
        }
        # No model_kwargs needed for GPT-2
    )
    return hf_llm


# Option 4: Phi-3 Mini (3.8B) - Good recent model
def load_phi3_cpu():
    """Load Phi-3 Mini optimized for CPU"""
    model_id = "microsoft/Phi-3-mini-4k-instruct"

    # Using dtype parameter
    hf_llm = HuggingFacePipeline.from_model_id(
        model_id=model_id,
        task="text-generation",
        device=-1,
        pipeline_kwargs={
            "max_new_tokens": 256,
            "temperature": 0.7,
            "do_sample": True,
        },
        model_kwargs={
            "dtype": torch.float32,  # Changed from torch_dtype
            "trust_remote_code": True,  # Sometimes needed for Phi-3
            "low_cpu_mem_usage": True,
        }
    )
    return hf_llm


# ============================================
# CPU-OPTIMIZED EMBEDDINGS MODEL
# ============================================

def get_cpu_embeddings():
    """Lightweight embeddings model for CPU"""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",  # Small, fast
        model_kwargs={'device': 'cpu'},  # Force CPU
        encode_kwargs={'normalize_embeddings': True}
    )


# ============================================
# COMPLETE RAG PIPELINE FOR CPU (UPDATED)
# ============================================

def create_cpu_rag_pipeline():
    """Create a complete RAG pipeline optimized for CPU"""

    # 1. Load a CPU-optimized model (choose one)
    print("Loading TinyLlama model on CPU...")
    llm = load_tinyllama_cpu()  # Best balance for CPU
    # llm = load_gpt2_cpu()  # Fastest option
    # llm = load_phi2_cpu()  # Better quality, slower
    # llm = load_phi3_cpu()  # Newer model

    # 2. Create prompt template
    prompt = ChatPromptTemplate.from_template("""
    You are a helpful AI assistant. Answer the question based on the provided context.

    Context: {context}

    Question: {input}

    Answer:
    """)

    # 3. Load or create documents (example)
    # For testing, let's create some sample documents
    sample_texts = [
        "Artificial intelligence (AI) is the simulation of human intelligence in machines.",
        "Machine learning is a subset of AI that enables systems to learn from data.",
        "Neural networks are computing systems inspired by biological neural networks.",
        "Deep learning uses multiple layers in neural networks to learn from data.",
        "Natural language processing (NLP) helps computers understand human language."
    ]

    # 4. Create embeddings and vector store on CPU
    print("Creating embeddings on CPU...")
    embeddings = get_cpu_embeddings()

    # Split texts into documents
    from langchain_core.documents import Document
    documents = [Document(page_content=text) for text in sample_texts]

    # Create vector store
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})  # Retrieve top 2 chunks

    # 5. Create chains
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

    return retrieval_chain, llm


# ============================================
# MAIN EXECUTION
# ============================================

def main():
    print("=" * 50)
    print("CPU-Only Local LLM with RAG")
    print("=" * 50)

    # Suppress the deprecation warning (optional)
    import warnings
    warnings.filterwarnings("ignore", message="`torch_dtype` is deprecated")

    # Create the pipeline
    rag_chain, llm = create_cpu_rag_pipeline()

    # Test questions
    questions = [
        "What is artificial intelligence?",
        "What is machine learning?",
        "Explain neural networks",
    ]

    # Run RAG queries
    print("\n" + "=" * 50)
    print("RAG-Enhanced Responses:")
    print("=" * 50)

    for question in questions:
        print(f"\nQuestion: {question}")
        print("-" * 30)

        response = rag_chain.invoke({
            "input": question
        })

        print(f"Answer: {response['answer']}")

        # Optional: Show retrieved context
        if 'context' in response:
            print("\nRetrieved Context:")
            for i, doc in enumerate(response['context'], 1):
                print(f"  {i}. {doc.page_content}")

    # Direct LLM usage (without RAG)
    print("\n" + "=" * 50)
    print("Direct LLM Responses (No RAG):")
    print("=" * 50)

    direct_questions = [
        "What is the capital of France?",
        "Explain quantum computing briefly."
    ]

    for question in direct_questions:
        print(f"\nQuestion: {question}")
        print("-" * 30)

        response = llm.invoke(question)
        print(f"Answer: {response}")


# ============================================
# LOAD FROM LOCAL PATH (UPDATED)
# ============================================

def load_from_local_path(model_path, tokenizer_path=None):
    """Load a model from a local directory (already downloaded)"""

    if tokenizer_path is None:
        tokenizer_path = model_path

    print(f"Loading model from local path: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float32,  # Changed from torch_dtype
        low_cpu_mem_usage=True
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.7,
    )

    hf_llm = HuggingFacePipeline(pipeline=pipe)
    return hf_llm


# ============================================
# QUICK TEST WITH MINIMAL CODE
# ============================================

def quick_test():
    """Minimal working example"""
    from transformers import pipeline
    from langchain_huggingface import HuggingFacePipeline
    from langchain_core.prompts import ChatPromptTemplate

    # Load model (notice no torch_dtype parameter)
    pipe = pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device=-1,
        max_new_tokens=100
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    # Simple chain
    prompt = ChatPromptTemplate.from_template("Question: {q}\nAnswer: ")
    chain = prompt | llm

    # Test
    result = chain.invoke({"q": "What is Python?"})
    print(f"Quick test result: {result}")


# ============================================
# RUN EVERYTHING
# ============================================

if __name__ == "__main__":
    # Either run the quick test
    quick_test()

    # Or run the full main pipeline
    # main()