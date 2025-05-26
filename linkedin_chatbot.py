import json
import os
import logging
import torch
import faiss
import streamlit as st
from typing import List, Dict, Any
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LinkedInChatBot:
    """
    A chatbot that generates comments for LinkedIn posts using a base model from Hugging Face
    and embeddings from a data.json file.
    """
    
    def __init__(
        self, 
        data_json_path: str = "Richard Persona.json",
        embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
        llm_model: str = 'facebook/opt-1.3b'  # Using a smaller, open-source model
    ):
        """
        Initialize the LinkedIn ChatBot with models and training data.
        
        Args:
            data_json_path: Path to the JSON file containing training data
            embedding_model: Model name for sentence embeddings
            llm_model: Model name for text generation
        """
        logger.info(f"Initializing LinkedInChatBot with {embedding_model} and {llm_model}")
        
        # Load embedding model
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
            logger.info(f"Successfully loaded embedding model: {embedding_model}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
        
        # Load language model with fallback options
        try:
            self.llm = pipeline("text-generation", model=llm_model)
            logger.info(f"Successfully loaded LLM model: {llm_model}")
        except Exception as e:
            logger.error(f"Failed to load primary LLM model {llm_model}: {e}")
            logger.info("Attempting to load fallback model: gpt2")
            try:
                self.llm = pipeline("text-generation", model="gpt2")
                logger.info("Successfully loaded fallback model: gpt2")
            except Exception as e2:
                logger.error(f"Failed to load fallback model: {e2}")
                raise Exception(f"Failed to load both primary ({llm_model}) and fallback (gpt2) models. Original error: {e}. Fallback error: {e2}")
        
        # Load training data
        self.messages = self._load_messages(data_json_path)
        logger.info(f"Loaded {len(self.messages)} messages from training data")
        
        # Generate embeddings for training data
        try:
            self.embeddings = self.embedding_model.encode(self.messages, convert_to_numpy=True)
            logger.info(f"Generated embeddings with shape {self.embeddings.shape}")
            
            # Create FAISS index for fast similarity search
            self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
            self.index.add(self.embeddings)
            logger.info("Successfully created FAISS index")
        except Exception as e:
            logger.error(f"Failed to create embeddings or FAISS index: {e}")
            raise
    
    def _load_messages(self, json_path: str) -> List[str]:
        """
        Load messages from the Richard Persona JSON file.
        
        Args:
            json_path: Path to the JSON file
            
        Returns:
            List of message content strings
        """
        logger.info(f"Loading messages from {json_path}")
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract all messages from the data
            messages = []
            for month, entries in data.items():
                for entry in entries:
                    for message in entry.get('messages', []):
                        if message.get('author') == 'ChatGPT':  # We want to learn from the AI responses
                            content = message.get('text', '').strip()
                            if content:
                                messages.append(content)
            
            logger.info(f"Successfully loaded {len(messages)} messages")
            return messages
        except Exception as e:
            logger.error(f"Failed to load messages: {e}")
            raise
    
    def generate_comment(self, post_text: str, k: int = 5, max_new_tokens: int = 250) -> str:
        """
        Generate a comment for the given LinkedIn post in the style of Richard.
        
        Args:
            post_text: The LinkedIn post text to comment on
            k: Number of similar messages to retrieve
            max_new_tokens: Maximum length of generated comment
            
        Returns:
            Generated comment text
        """
        logger.info(f"Generating comment for post: {post_text[:50]}...")
        
        try:
            # Encode post text
            post_emb = self.embedding_model.encode([post_text])
            
            # Find similar messages
            D, I = self.index.search(post_emb, k)
            similar_msgs = [self.messages[i] for i in I[0]]
            logger.info(f"Found {len(similar_msgs)} similar messages")
            
            # Create prompt for the LLM
            prompt = (
                "You are Richard, a professional LinkedIn commenter with a specific analytical and insightful style. "
                "Your comments are well-structured, often using bullet points, and provide thoughtful analysis. "
                "Your task is to write a comment on the following LinkedIn post that matches your distinctive style.\n\n"
                "Here are examples of your previous comments that show your style and tone:\n"
                + "\n".join([f"EXAMPLE {i+1}:\n{msg[:300]}...\n" for i, msg in enumerate(similar_msgs)])
                + f"\n\nLinkedIn Post to comment on:\n{post_text}\n\n"
                "Write your comment in your distinctive style:"
            )
            
            # Generate response
            result = self.llm(prompt, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7)
            generated_text = result[0]['generated_text']
            
            # Extract only the comment part
            if "Write your comment in your distinctive style:" in generated_text:
                comment = generated_text.split("Write your comment in your distinctive style:", 1)[1].strip()
            else:
                # Extract text after the prompt
                prompt_end = prompt.strip()[-50:]  # Last 50 chars of prompt for reliable splitting
                if prompt_end in generated_text:
                    comment = generated_text.split(prompt_end, 1)[1].strip()
                else:
                    comment = generated_text.replace(prompt, "").strip()
            
            logger.info(f"Successfully generated comment: {comment[:50]}...")
            return comment
        except Exception as e:
            logger.error(f"Failed to generate comment: {e}")
            raise

def create_streamlit_app():
    """
    Create a Streamlit app for the LinkedIn ChatBot.
    """
    st.set_page_config(page_title="LinkedIn Comment Generator", page_icon="💬", layout="wide")
    
    st.title("LinkedIn Post Comment Generator")
    st.markdown("""
    This app generates professional comments for LinkedIn posts using AI. 
    Enter a LinkedIn post below, and the AI will generate a thoughtful comment based on the content.
    """)
    
    # Sidebar for model selection
    st.sidebar.title("Model Settings")
    embedding_model = st.sidebar.selectbox(
        "Embedding Model",
        ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2"],
        index=0
    )
    
    llm_model = st.sidebar.selectbox(
        "Language Model",
        ["mistralai/Mistral-7B-Instruct-v0.2", "google/flan-t5-base", "gpt2"],
        index=0
    )
    
    # Initialize session state
    if 'chatbot' not in st.session_state:
        with st.spinner("Loading models... This may take a few minutes."):
            try:
                st.session_state.chatbot = LinkedInChatBot(
                    embedding_model=embedding_model,
                    llm_model=llm_model
                )
                st.success("Models loaded successfully!")
            except Exception as e:
                st.error(f"Error loading models: {str(e)}")
                st.stop()
    
    # Input area for LinkedIn post
    post_text = st.text_area("Enter LinkedIn Post", height=200)
    
    # Generate comment button
    if st.button("Generate Comment"):
        if not post_text:
            st.warning("Please enter a LinkedIn post.")
        else:
            with st.spinner("Generating comment..."):
                try:
                    comment = st.session_state.chatbot.generate_comment(post_text)
                    st.subheader("Generated Comment:")
                    st.write(comment)
                    
                    # Copy button
                    st.text_area("Copy this comment:", value=comment, height=150)
                    if st.button("Copy to Clipboard"):
                        st.write("Comment copied to clipboard!")
                except Exception as e:
                    st.error(f"Error generating comment: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("Powered by Hugging Face models and trained on professional LinkedIn interactions.")

if __name__ == "__main__":
    create_streamlit_app()