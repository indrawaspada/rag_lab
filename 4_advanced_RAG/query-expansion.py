import os
from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

class ParaphrasedQuery(BaseModel):
    """You have performed query expansion to generate a paraphrasing of a question."""
    
    paraphrased_query: str = Field(
        ...,
        description="A unique paraphrasing of the original question.",
    )

class QueryExpander:
    """A class to handle query expansion using LangChain and OpenAI."""
    
    def __init__(self, api_key: str = None):
        """Initialize the QueryExpander with necessary components."""
        # Set up OpenAI API key
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        elif not os.getenv("OPENAI_API_KEY"):
            raise ValueError("No API key provided. Set OPENAI_API_KEY environment variable or pass key to constructor.")
        
        # Define the system prompt
        self.system_prompt = """You are an expert at expanding user questions into multiple variations. \
            Perform query expansion. If there are multiple common ways of phrasing a user question \
            or common synonyms for key words in the question, make sure to return multiple versions \
            of the query with the different phrasings.

            If there are acronyms or words you are not familiar with, do not try to rephrase them.

            Return at least 3 versions of the question that maintain the original intent."""
        
        # Create parser for JSON output
        self.parser = JsonOutputParser(pydantic_object=ParaphrasedQuery)
        
        # Set up the prompt template with format instructions
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{question}\n\n{format_instructions}")
        ])
        
        # Initialize the language model
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0
        )
        
        # Create the chain
        self.query_analyzer = (
            self.prompt.partial(format_instructions=self.parser.get_format_instructions())
            | self.llm 
            | self.parser
        )
    
    def expand_query(self, question: str) -> List[str]:
        """
        Expand a question into multiple paraphrased variations.
        
        Args:
            question (str): The original question to expand
            
        Returns:
            List[str]: List of paraphrased variations of the question
        """
        try:
            # Get paraphrased query
            result = self.query_analyzer.invoke({"question": question})
            
            # The result should be a ParaphrasedQuery object with paraphrased_query field
            if isinstance(result, dict):
                variation = result.get("paraphrased_query", "")
            else:
                variation = result.paraphrased_query
            
            # Return as a list with the variation
            return [variation] if variation else []
            
            return variations
            
        except Exception as e:
            print(f"Error expanding query: {str(e)}")
            return []

def main():
    """Example usage of the QueryExpander"""
    try:
        # Initialize the expander
        expander = QueryExpander()
        
        print("Welcome to LangChain Query Expander!")
        print("Enter a question to see different variations (or 'quit' to exit)")
        print("\nExample questions:")
        print("- How to use multi-modal models in a chain?")
        print("- What's the best way to stream events from an LLM agent?")
        print("- How to implement RAG with vector databases?")
        
        while True:
            question = input("\nEnter your question: ").strip()
            
            if question.lower() == 'quit':
                print("Thank you for using LangChain Query Expander. Goodbye!")
                break
                
            print("\nGenerating variations...")
            variations = expander.expand_query(question)
            
            print("\nExpanded Queries:")
            for i, variation in enumerate(variations, 1):
                print(f"\n{i}. {variation}")
            
            print("\nTotal variations generated:", len(variations))
            
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Please check your API key and try again.")

if __name__ == "__main__":
    main()