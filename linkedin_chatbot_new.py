import os
import json
import logging
import torch
import faiss
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import spacy
import re
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('linkedin_chatbot')

class LinkedInChatBot:
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
        Load messages from a JSON file.
        
        Args:
            json_path: Path to the JSON file containing messages
            
        Returns:
            List of messages
        """
        logger.info(f"Loading messages from {json_path}")
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            messages = []
            # Robustly extract all 'text' or 'Comment' fields from any list of dicts
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        if 'text' in item and item['text']:
                            messages.append(item['text'])
                        elif 'Comment' in item and item['Comment']:
                            messages.append(item['Comment'])
            # Check if it's the Richard format
            elif 'messages' in data:
                logger.info(f"Detected Richard format in {json_path}")
                for msg in data['messages']:
                    if 'content' in msg and msg['content']:
                        messages.append(msg['content'])
            # Unknown format
            else:
                logger.warning(f"Unknown format in {json_path}, attempting to extract text")
                def extract_text(obj):
                    if isinstance(obj, dict):
                        for k, v in obj.items():
                            if k in ['text', 'content', 'message'] and isinstance(v, str) and v:
                                messages.append(v)
                            elif isinstance(v, (dict, list)):
                                extract_text(v)
                    elif isinstance(obj, list):
                        for item in obj:
                            extract_text(item)
                extract_text(data)
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
            
            # Extract key topics from the post
            topics = self._extract_topics(post_text)
            
            # --- Persona/Style separation: Train on Richard, comment/test on Skellett ---
            # Load Richard Persona for persona learning (training)
            richard_msgs = []
            try:
                with open('Richard Persona.json', 'r', encoding='utf-8') as f:
                    richard_data = json.load(f)
                    if 'messages' in richard_data:
                        richard_msgs = [msg['content'] for msg in richard_data['messages'] if 'content' in msg and msg['content']]
            except Exception as e:
                logger.warning(f"Could not load Richard Persona for persona learning: {e}")

            # Load Skellett for comment style/testing
            skellett_msgs = []
            try:
                with open('Skellett.json', 'r', encoding='utf-8') as f:
                    skellett_data = json.load(f)
                    if isinstance(skellett_data, list) and 'text' in skellett_data[0]:
                        skellett_msgs = [item['text'] for item in skellett_data if 'text' in item and item['text']]
            except Exception as e:
                logger.warning(f"Could not load Skellett for comment style: {e}")

            # Use Richard for persona (embedding, analytical tone), Skellett for comment style/length
            all_style_msgs = skellett_msgs
            persona_msgs = richard_msgs
            # Use embedding model to get closest Skellett style messages for the post
            if skellett_msgs:
                try:
                    skellett_embs = self.embedding_model.encode(skellett_msgs, convert_to_numpy=True)
                    D_style = faiss.IndexFlatL2(skellett_embs.shape[1])
                    D_style.add(skellett_embs)
                    _, I_style = D_style.search(post_emb, min(8, len(skellett_msgs)))
                    fused_msgs = [skellett_msgs[i] for i in I_style[0]]
                except Exception as e:
                    logger.warning(f"Could not compute Skellett style messages: {e}")
                    fused_msgs = similar_msgs
            else:
                fused_msgs = similar_msgs

            # --- Prompt construction: Skellett for comment, Richard for persona ---
            import random
            random.shuffle(fused_msgs)
            fused_examples = "\n".join([f"- {msg}" for msg in fused_msgs[:5]])
            topics_str = ", ".join(topics) if topics else "No specific topics identified"
            prompt = f"""
You are an analytical professional who writes in the style of Richard (see persona data), but your comment structure and length should match the Skellett examples below.

POST:
{post_text}

KEY TOPICS:
{topics_str}

SKELLETT STYLE EXAMPLES:
{fused_examples}

Generate a unique, insightful, and meaningful comment that addresses the post's key topics. The comment must be completely new and not appear in any of the reference examples or training data. Do not copy, paraphrase, or reuse any example. Use your own words and style. The comment should be concise, professional, engaging, and add genuine value or perspective to the conversation. Match the length and structure of the Skellett persona, but blend in the analytical tone of Richard.
"""
            # Generate comment using the language model
            result = self.llm(prompt, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.92, top_p=0.90)
            generated_text = result[0]['generated_text']
            comment = generated_text[len(prompt):].strip()

            # Fallback logic: ensure uniqueness (not in any persona JSON)
            def is_unique_comment(cmt, persona_msgs, style_msgs):
                cmt_norm = cmt.strip().lower()
                for msg in persona_msgs + style_msgs:
                    if cmt_norm == msg.strip().lower() or cmt_norm in msg.strip().lower() or msg.strip().lower() in cmt_norm:
                        return False
                return True

            persona_msgs_norm = [m.strip().lower() for m in persona_msgs]
            style_msgs_norm = [m.strip().lower() for m in all_style_msgs]
            tries = 0
            while (len(comment) < 10 or not is_unique_comment(comment, persona_msgs_norm, style_msgs_norm)) and tries < 3:
                # Regenerate with a slightly different seed
                result = self.llm(prompt, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.92 + 0.02*tries, top_p=0.90 - 0.02*tries)
                generated_text = result[0]['generated_text']
                comment = generated_text[len(prompt):].strip()
                tries += 1
            if len(comment) < 10 or not is_unique_comment(comment, persona_msgs_norm, style_msgs_norm):
                import uuid
                comment = f"[AI-{uuid.uuid4().hex[:6]}] {post_text[:40]}... This is a new perspective on {', '.join(topics) if topics else 'the topic'}."

            # --- Length control: match Skellett style length (approx 20-40 words) ---
            words = comment.split()
            if len(words) > 45:
                comment = ' '.join(words[:42]) + '...'
            elif len(words) < 15 and len(fused_msgs) > 0:
                # If too short, append a relevant phrase from Skellett style
                comment += ' ' + fused_msgs[0][:60]

            logger.info(f"Successfully generated comment: {comment[:50]}...")
            return comment
        except Exception as e:
            logger.error(f"Failed to generate comment: {e}")
            raise
    
    def _extract_topics(self, text: str) -> list:
        """
        Extract key topics from the given text.
        
        Args:
            text: The text to extract topics from
            
        Returns:
            List of key topics
        """
        # Simple keyword extraction based on noun phrases
        try:
            # Try to use spaCy if available
            import spacy
            try:
                nlp = spacy.load("en_core_web_sm")
            except:
                # If the model isn't available, download it
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
                nlp = spacy.load("en_core_web_sm")
            
            doc = nlp(text)
            
            # Extract noun phrases and named entities
            topics = []
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) <= 3:  # Limit to phrases with 3 or fewer words
                    topics.append(chunk.text)
            
            for ent in doc.ents:
                if ent.label_ in ['ORG', 'PRODUCT', 'GPE', 'PERSON', 'NORP']:
                    topics.append(ent.text)
            
            # Clean up and deduplicate
            clean_topics = []
            for topic in topics:
                # Remove stopwords and punctuation
                clean_topic = re.sub(r'[^\w\s]', '', topic).strip()
                if clean_topic and len(clean_topic) > 2:
                    clean_topics.append(clean_topic)
            
            # Count occurrences and get the most common topics
            topic_counter = Counter(clean_topics)
            common_topics = [topic for topic, count in topic_counter.most_common(5)]
            
            return common_topics
            
        except Exception as e:
            logger.warning(f"Failed to extract topics using spaCy: {e}")
            
            # Fallback to simple word frequency
            words = re.findall(r'\b[A-Z][a-z]+\b', text)  # Find capitalized words as potential topics
            if not words:
                words = re.findall(r'\b[a-zA-Z]{4,}\b', text)  # Find words with 4+ characters
            
            # Count occurrences and get the most common words
            word_counter = Counter(words)
            common_words = [word for word, count in word_counter.most_common(5)]
            
            return common_words

def create_streamlit_app():
    """
    Create a Streamlit app for the LinkedIn ChatBot.
    """
    import streamlit as st
    
    st.title("LinkedIn Comment Generator")
    st.write("Generate comments for LinkedIn posts in a specific style")
    
    # Sidebar for model selection
    st.sidebar.header("Model Settings")
    embedding_model = st.sidebar.selectbox(
        "Embedding Model",
        ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"],
        index=0
    )
    
    llm_model = st.sidebar.selectbox(
        "Language Model",
        ["facebook/opt-1.3b", "gpt2", "EleutherAI/gpt-neo-125M"],
        index=0
    )
    
    data_json = st.sidebar.selectbox(
        "Training Data",
        ["Skellett.json", "Richard Persona.json"],
        index=0
    )
    
    # Initialize the chatbot
    @st.cache_resource
    def load_chatbot(data_json, embedding_model, llm_model):
        return LinkedInChatBot(
            data_json_path=data_json,
            embedding_model=embedding_model,
            llm_model=llm_model
        )
    
    with st.spinner("Loading models and data..."):
        try:
            chatbot = load_chatbot(data_json, embedding_model, llm_model)
            st.success("Models loaded successfully!")
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            st.stop()
    
    # Input for LinkedIn post
    st.header("LinkedIn Post")
    post_text = st.text_area("Enter the LinkedIn post text:", height=150)
    
    # Generate comment
    if st.button("Generate Comment"):
        if not post_text:
            st.warning("Please enter a LinkedIn post.")
        else:
            with st.spinner("Generating comment..."):
                try:
                    comment = chatbot.generate_comment(post_text)
                    st.header("Generated Comment")
                    st.write(comment)
                    
                    # Copy button
                    st.code(comment)
                    st.button("Copy to clipboard", key="copy")
                except Exception as e:
                    st.error(f"Error generating comment: {str(e)}")
    
    # About section
    st.sidebar.header("About")
    st.sidebar.info(
        "This app uses AI to generate comments for LinkedIn posts in a specific style. "
        "It uses sentence embeddings to find similar messages in the training data, "
        "and a language model to generate a comment based on the post and similar messages."
    )

if __name__ == "__main__":
    create_streamlit_app()