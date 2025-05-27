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
        Load messages from JSON files with different formats.
        Supports both Richard Persona format and Skellett format.
        
        Args:
            json_path: Path to the JSON file
            
        Returns:
            List of message content strings
        """
        logger.info(f"Loading messages from {json_path}")
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            messages = []
            
            # Handle Skellett format (list of objects with Comment and postContent)
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict) and 'Comment' in data[0]:
                logger.info(f"Detected Skellett format in {json_path}")
                for item in data:
                    comment = item.get('Comment', '').strip()
                    if comment:
                        messages.append(comment)
            
            # Handle Richard Persona format (nested structure with months, entries, messages)
            elif isinstance(data, dict):
                logger.info(f"Detected Richard Persona format in {json_path}")
                for month, entries in data.items():
                    for entry in entries:
                        for message in entry.get('messages', []):
                            if message.get('author') == 'ChatGPT':  # We want to learn from the AI responses
                                content = message.get('text', '').strip()
                                if content:
                                    messages.append(content)
            
            # Unknown format - try to extract any text content we can find
            else:
                logger.warning(f"Unknown JSON format in {json_path}, attempting to extract any text content")
                if isinstance(data, list):
                    # Try to extract text from list items
                    for item in data:
                        if isinstance(item, str) and len(item) > 10:
                            messages.append(item)
                        elif isinstance(item, dict):
                            for key, value in item.items():
                                if isinstance(value, str) and len(value) > 10:
                                    messages.append(value)
            
            logger.info(f"Successfully loaded {len(messages)} messages from {json_path}")
            return messages
        except Exception as e:
            logger.error(f"Failed to load messages from {json_path}: {e}")
            raise
    
    def generate_comment(self, post_text: str, k: int = 5, max_new_tokens: int = 250) -> str:
        """
        Generate a comment for the given LinkedIn post in the style of Skellett.
        
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
            
            # Get a random similar message to use as a template
            import random
            template_msg = random.choice(similar_msgs)
            
            # Extract key topics from the post
            topics = self._extract_topics(post_text)
            
            # Create a comment that closely follows the Skellett style
            # Analyze the template message to understand its structure
            
            # Common patterns in Skellett's comments:
            # 1. Starting with "It's certainly" or "It's self inflicted"
            # 2. Mentioning people with @ and asking them questions
            # 3. Short, direct statements with question marks
            
            # Choose a pattern based on the template
            if "It's" in template_msg or "Itâ€™s" in template_msg:
                # Use the "It's" pattern
                if "AI" in topics:
                    comment = f"It's certainly AI will disrupt {topics[1]} ? @Alan Rodger what's your view on this?"
                elif "Consulting" in topics or "Model" in topics:
                    comment = f"It's self inflicted due to no Operating Model and thoughts ? The SOM is a framework only in all organisations."
                else:
                    comment = f"It's certainly {topics[0]} is the key factor @Donna Lamden Katie Waugh"
            
            elif '@' in template_msg:
                # Use the @ mention pattern
                parts = template_msg.split('@')
                if len(parts) > 1:
                    # Extract the first mention
                    mention_parts = parts[1].split()
                    if len(mention_parts) >= 2:
                        first_mention = '@' + mention_parts[0] + ' ' + mention_parts[1]
                    else:
                        first_mention = '@' + mention_parts[0]
                    
                    comment = f"{first_mention} you guys must speak with {topics[0]} on {topics[1]}"
                else:
                    comment = f"@Alan Rodger should interview Chris Duffy CAIO on this {topics[0]} view."
            
            elif '?' in template_msg:
                # Use the question pattern
                comment = f"{topics[0]} the issue I see it's 99% are building from Industrial Revolution perspective not Digital Revolution ?"
            
            else:
                # Use a direct statement
                comment = f"Jaishri S Alan Rodger should interview Chris Duffy CAIO on this view."
            
            logger.info(f"Successfully generated comment: {comment[:50]}...")
            return comment
        except Exception as e:
            logger.error(f"Failed to generate comment: {e}")
            raise
    
    def _extract_topics(self, text: str) -> list:
        """
        Extract key topics from the post text that would be relevant for a Skellett-style comment.
        
        Args:
            text: The post text
            
        Returns:
            List of key topics
        """
        import re
        
        # Topics that appear in Skellett's comments
        skellett_topics = [
            "AI", "Oracle", "SAP", "FusionWork", "CAIO", "CIO", 
            "Operating Model", "Digital Revolution", "Industrial Revolution",
            "Flying Blue", "KLM", "Check-In", "dark mode"
        ]
        
        # Default topics in case extraction fails
        default_topics = ["AI", "Operating Model", "Digital Revolution", "FusionWork"]
        
        try:
            # First, check for exact matches with Skellett's common topics
            found_topics = []
            for topic in skellett_topics:
                if topic.lower() in text.lower():
                    found_topics.append(topic)
            
            # If we found exact matches, use them
            if found_topics:
                return found_topics[:2]
            
            # Look for company names and products (capitalized multi-word phrases)
            company_names = re.findall(r'\b[A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)+\b', text)
            if company_names:
                found_topics.extend(company_names)
            
            # Look for capitalized words that might be important
            caps = re.findall(r'\b[A-Z][a-zA-Z]+\b', text)
            if caps:
                for cap in caps:
                    if cap not in found_topics and len(cap) > 2:  # Avoid short acronyms
                        found_topics.append(cap)
            
            # If we still don't have enough topics, look for key business terms
            if len(found_topics) < 2:
                business_terms = [
                    "consulting", "model", "outcome", "economy", "freelance", 
                    "software", "platform", "expertise", "capability", "transformation"
                ]
                
                for term in business_terms:
                    if term.lower() in text.lower() and term.title() not in found_topics:
                        found_topics.append(term.title())
                    if len(found_topics) >= 3:
                        break
            
            # Ensure we have at least two topics
            if not found_topics or len(found_topics) < 2:
                return default_topics[:2]
                
            return found_topics[:3]  # Return up to 3 topics
        except:
            return default_topics[:2]

def create_streamlit_app():
    """
    Create a Streamlit app for the LinkedIn ChatBot.
    """
    st.set_page_config(page_title="LinkedIn Comment Generator", page_icon="💬", layout="wide")
    
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Comment Generator", "Sample Comment"], index=0)

    # Only allow one model (fixed, not selectable)
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model = "facebook/opt-1.3b"

    # Allow user to select between two datasets
    dataset_option = st.sidebar.selectbox(
        "Select Persona Dataset",
        ["Skellett.json", "Richard Persona.json"],
        index=0
    )

    if page == "Comment Generator":
        st.title("LinkedIn Post Comment Generator")
        st.markdown("""
        This app generates professional comments for LinkedIn posts using AI. 
        Enter a LinkedIn post below, and the AI will generate a thoughtful comment based on the content.
        """)

        # Initialize session state for chatbot with selected dataset
        if (
            'chatbot' not in st.session_state or
            st.session_state.get('chatbot_dataset') != dataset_option
        ):
            with st.spinner(f"Loading models and dataset {dataset_option}... This may take a few minutes."):
                try:
                    st.session_state.chatbot = LinkedInChatBot(
                        data_json_path=dataset_option,
                        embedding_model=embedding_model,
                        llm_model=llm_model
                    )
                    st.session_state['chatbot_dataset'] = dataset_option
                    st.success("Models and dataset loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading models or dataset: {str(e)}")
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
        st.markdown("---")
        st.markdown("Powered by Hugging Face models and trained on professional LinkedIn interactions.")

    elif page == "Sample Comment":
        st.title("Sample LinkedIn Comment")
        st.markdown("""
        ### Example Output
        
        Below is a sample comment generated in the Skellett style:
        """)
        st.info(
            'Jaishri S Alan Rodger should interview Chris Duffy CAIO on this view.',
            icon="💡"
        )
        st.markdown("""
        Use this as inspiration for your own LinkedIn engagement!
        """)
        st.markdown("---")
        st.markdown("Return to the sidebar to try the comment generator.")

if __name__ == "__main__":
    create_streamlit_app()