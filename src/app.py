import os
from typing import List
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from vectordb import VectorDB
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader

# Load environment variables
load_dotenv()


def load_documents() -> List[str]:
    """
    Load documents for demonstration.

    Returns:
        List of sample documents
    """
    results = []
    documents_path = "./data"
    for file in os.listdir(documents_path):
        file_path = os.path.join(documents_path, file)
        try:
            loader = TextLoader(file_path)
            loaded_docs = loader.load()
            results.extend(loaded_docs)
            print(f"Successfully loaded: {file}")
        except Exception as e:
            print(f"Error loading {file}: {str(e)}")

    return results


class RAGAssistant:
    """
    A simple RAG-based AI assistant using ChromaDB and multiple LLM providers.
    Supports OpenAI, Groq, and Google Gemini APIs.
    """

    def __init__(self):
        """Initialize the RAG assistant."""
        # Initialize LLM - check for available API keys in order of preference
        self.llm = self._initialize_llm()
        if not self.llm:
            raise ValueError(
                "No valid API key found. Please set one of: "
                "OPENAI_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY in your .env file"
            )

        self.chat_history = []

        # Initialize vector database
        self.vector_db = VectorDB()

        # Create RAG prompt template
        self.prompt_template = ChatPromptTemplate.from_template(
            template="""
            
            You are a helpful {role}. 
            
            Context:
            {context}
            
            Instructions:
            {instructions}


            Reasoning Process:
            {reasoning_process}

            Output Constraints:
            {output_constraints}

            Style or Tone:
            {style_or_tone}

            Goal:
            {goal}

            Researcher's Question:
            {question}
            """
            )

        # Create the chain
        self.chain = self.prompt_template | self.llm | StrOutputParser()

        print("RAG Assistant initialized successfully")

    def _initialize_llm(self):
        """
        Initialize the LLM by checking for available API keys.
        Tries OpenAI, Groq, and Google Gemini in that order.
        """
        # Check for OpenAI API key
        if os.getenv("OPENAI_API_KEY"):
            model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            print(f"Using OpenAI model: {model_name}")
            return ChatOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"), model=model_name, temperature=0.0
            )

        elif os.getenv("GROQ_API_KEY"):
            model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
            print(f"Using Groq model: {model_name}")
            return ChatGroq(
                api_key=os.getenv("GROQ_API_KEY"), model=model_name, temperature=0.0
            )

        elif os.getenv("GOOGLE_API_KEY"):
            model_name = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
            print(f"Using Google Gemini model: {model_name}")
            return ChatGoogleGenerativeAI(
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                model=model_name,
                temperature=0.0,
            )

        else:
            raise ValueError(
                "No valid API key found. Please set one of: OPENAI_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY in your .env file"
            )

    def add_documents(self, documents: List) -> None:
        """
        Add documents to the knowledge base.

        Args:
            documents: List of documents
        """
        self.vector_db.add_documents(documents)

    def invoke(self, input: str, n_results: int = 3) -> str:
        """
        Query the RAG assistant.

        Args:
            input: User's input
            n_results: Number of relevant chunks to retrieve

        Returns:
            Dictionary containing the answer and retrieved context
        """
        llm_answer = ""
        results = self.vector_db.search(input)

        # Format results
        relevant_chunks = []
        for i, doc in enumerate(results["documents"][0]):
            relevant_chunks.append({
                "content": doc,
                "title": results["metadatas"][0][i]["source"],
                "similarity": 1 - results["distances"][0][i]  # Convert distance to similarity
            })

        # Add chat history to the prompt
        chat_history = "\n\n".join([
            f"{chat['role']}: {chat['content']}"
            for chat in self.chat_history[-10:]
        ])

        # Build context from research
        relevant_context = "\n\n".join([
            f"From {chunk['title']}:\n{chunk['content']}" 
            for chunk in relevant_chunks
        ])

        # Prompt Builder Config
        role = "Research Assistant"
        context = f"""
            Relevant Context:
            {relevant_context}

            Chat History:
            {chat_history}
        """
        instructions = "You are tasked with answering questions based on the research findings."
        reasoning_process = "Chain of Thought"
        output_constraints = "The output should have 2 clear sections: namely reasoning and answer."
        style_or_tone = "Plain and concise."
        goal = "Provide a comprehensive answer based on the research findings below."


        print(f"{len(results)} relevant chunks found.")

        # Generate answer
        llm_answer = self.chain.invoke({"context":context, "question":input, "reasoning_process":reasoning_process, "instructions":instructions, "role":role, "output_constraints":output_constraints, "style_or_tone":style_or_tone, "goal":goal})

        return llm_answer


def main():
    """Main function to demonstrate the RAG assistant."""
    try:
        # Initialize the RAG assistant
        print("Initializing RAG Assistant...")
        assistant = RAGAssistant()

        # Load sample documents
        print("\nLoading documents...")
        sample_docs = load_documents()
        print(f"Loaded {len(sample_docs)} sample documents")

        assistant.add_documents(sample_docs)

        done = False

        while not done:
            question = input("Enter a question or 'quit' to exit: ")
            if question.lower() == "quit":
                done = True
            else:
                result = assistant.invoke(question)
                assistant.chat_history.append({"role": "user", "content": question})
                assistant.chat_history.append({"role": "assistant", "content": result})
                print("Answer:\n", result)

    except Exception as e:
        print(f"Error running RAG assistant: {e}")
        print("Make sure you have set up your .env file with at least one API key:")
        print("- OPENAI_API_KEY (OpenAI GPT models)")
        print("- GROQ_API_KEY (Groq Llama models)")
        print("- GOOGLE_API_KEY (Google Gemini models)")


if __name__ == "__main__":
    main()
