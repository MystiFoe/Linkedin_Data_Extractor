import os
import json
import logging
import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from typing import List, Dict, Any, Optional
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("persona_generator.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class PersonaCommentGenerator:
    """
    A class to generate LinkedIn comments in the style of a specific persona
    based on their historical messages.
    """
    
    def __init__(
        self, 
        persona_json_path: str, 
        embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
        llm_model: str = 'mistralai/Mistral-7B-Instruct-v0.2'
    ):
        """
        Initialize the PersonaCommentGenerator with models and persona data.
        
        Args:
            persona_json_path: Path to the JSON file containing persona messages
            embedding_model: Model name for sentence embeddings
            llm_model: Model name for text generation
        """
        logger.info(f"Initializing PersonaCommentGenerator with {embedding_model} and {llm_model}")
        
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
            logger.info(f"Successfully loaded embedding model: {embedding_model}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
            
        try:
            self.llm = pipeline("text-generation", model=llm_model)
            logger.info(f"Successfully loaded LLM model: {llm_model}")
        except Exception as e:
            logger.error(f"Failed to load LLM model: {e}")
            raise
            
        self.messages = self._load_messages(persona_json_path)
        logger.info(f"Loaded {len(self.messages)} messages from persona data")
        
        # Generate embeddings for persona messages
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
        Load messages from the persona JSON file.
        
        Args:
            json_path: Path to the JSON file
            
        Returns:
            List of message content strings
        """
        logger.info(f"Loading messages from {json_path}")
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Flatten all messages from all months
            messages = []
            for month in data:
                for msg in month.get('messages', []):
                    content = msg.get('content', '').strip()
                    if content:
                        messages.append(content)
                        
            logger.info(f"Successfully loaded {len(messages)} messages")
            return messages
        except Exception as e:
            logger.error(f"Failed to load messages: {e}")
            raise

    def generate_comment(self, post_text: str, k: int = 5, max_new_tokens: int = 150) -> str:
        """
        Generate a comment in the persona's style for the given post.
        
        Args:
            post_text: The LinkedIn post text to comment on
            k: Number of similar messages to retrieve
            max_new_tokens: Maximum length of generated comment
            
        Returns:
            Generated comment text
        """
        logger.info(f"Generating comment for post: {post_text[:50]}...")
        
        try:
            # Extract just the post text if it contains additional instructions
            post_lines = post_text.split("\n")
            actual_post = post_text
            if "Post:" in post_text:
                for line in post_lines:
                    if line.startswith("Post:"):
                        actual_post = line.replace("Post:", "").strip()
                        # Get any following lines that might be part of the post
                        post_index = post_lines.index(line)
                        if post_index + 1 < len(post_lines) and post_lines[post_index + 1].strip() and not post_lines[post_index + 1].startswith("Write"):
                            actual_post += " " + post_lines[post_index + 1].strip()
            
            # Encode post text
            post_emb = self.embedding_model.encode([actual_post])
            
            # Find similar messages
            D, I = self.index.search(post_emb, k)
            similar_msgs = [self.messages[i] for i in I[0]]
            logger.info(f"Found {len(similar_msgs)} similar messages")
            
            # Create prompt for the LLM
            prompt = (
                "You are a LinkedIn professional with a specific writing style. Your task is to write an authentic comment "
                "on a LinkedIn post that perfectly matches the style shown in these example comments from your history.\n\n"
                "Here are examples of your past comments:\n"
                + "\n".join([f"- {msg}" for msg in similar_msgs])
                + f"\n\n{post_text}\n\n"
            )
            
            # Generate response
            result = self.llm(prompt, max_new_tokens=max_new_tokens)
            generated_text = result[0]['generated_text']
            
            # Extract only the comment part
            if "Comment:" in generated_text:
                comment = generated_text.split("Comment:", 1)[1].strip()
            elif "Write an authentic comment:" in generated_text:
                comment = generated_text.split("Write an authentic comment:", 1)[1].strip()
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

class PersonaManagerModule:
    """
    Module to manage persona data and model files.
    """
    
    def __init__(self, base_dir: str = "persona_models"):
        """
        Initialize the PersonaManagerModule.
        
        Args:
            base_dir: Base directory for storing persona data
        """
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        
        # Add default path for Richard persona
        self.default_persona_path = r"C:\Users\m.giritharan\linkedin_data_extractor\Richard Persona.json"
        if os.path.exists(self.default_persona_path):
            logger.info(f"Found default Richard persona at {self.default_persona_path}")
        else:
            logger.warning(f"Default Richard persona not found at {self.default_persona_path}")
        
    def list_personas(self) -> List[str]:
        """
        List available personas.
        
        Returns:
            List of persona names
        """
        personas = []
        
        # Add default Richard persona if it exists
        if os.path.exists(self.default_persona_path):
            personas.append("Richard (Default)")
            
        try:
            # Add any other personas from the personas directory
            file_personas = [f.replace('.json', '') for f in os.listdir(self.base_dir) 
                    if f.endswith('.json')]
            personas.extend(file_personas)
            return personas
        except Exception as e:
            logger.error(f"Failed to list personas: {e}")
            if personas:  # At least return Richard if available
                return personas
            return []
            
    def get_persona_path(self, persona_name: str) -> str:
        """
        Get the file path for a persona.
        
        Args:
            persona_name: Name of the persona
            
        Returns:
            Path to the persona JSON file
        """
        # Check if it's the default Richard persona
        if persona_name == "Richard (Default)":
            return self.default_persona_path
            
        # Otherwise, look in the personas directory
        return os.path.join(self.base_dir, f"{persona_name}.json")
        
    def save_uploaded_persona(self, uploaded_file, persona_name: str) -> str:
        """
        Save an uploaded persona file.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            persona_name: Name to save the persona as
            
        Returns:
            Path to the saved file
        """
        try:
            file_path = self.get_persona_path(persona_name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            logger.info(f"Saved persona {persona_name} to {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Failed to save persona: {e}")
            raise

def persona_comment_generator_page():
    """
    Streamlit page for the Persona Comment Generator feature.
    """
    st.title("🎭 LinkedIn Persona Comment Generator")
    
    # Initialize manager
    persona_manager = PersonaManagerModule()
    
    # Check if default persona exists
    has_default_persona = os.path.exists(persona_manager.default_persona_path)
    
    st.write(
        "Generate LinkedIn comments in the style of a specific persona based on their historical messages. "
        "The AI will analyze the writing patterns and generate comments that match the persona's tone and style."
    )
    
    if has_default_persona:
        st.success("✓ Richard's persona data successfully loaded")
    
    # UI Sections using tabs
    tab1, tab2 = st.tabs(["Generate Comments", "Manage Personas"])
    
    # Tab 1: Generate Comments
    with tab1:
        st.header("Generate Persona-Based Comments")
        
        # Get available personas
        personas = persona_manager.list_personas()
        
        if not personas:
            st.warning("No personas available. Please upload a persona JSON file in the 'Manage Personas' tab.")
        else:
            # Persona selection
            col1, col2 = st.columns([2, 1])
            
            with col1:
                selected_persona = st.selectbox(
                    "Select persona", 
                    options=personas,
                    index=0 if "Richard (Default)" in personas else 0,
                    help="Choose the persona whose style you want to mimic"
                )
            
            with col2:
                if st.button("📊 View Persona Stats", use_container_width=True):
                    try:
                        persona_path = persona_manager.get_persona_path(selected_persona)
                        with open(persona_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        # Count messages
                        message_count = sum(len(month.get('messages', [])) for month in data)
                        
                        # Calculate average message length
                        all_messages = []
                        for month in data:
                            for msg in month.get('messages', []):
                                content = msg.get('content', '').strip()
                                if content:
                                    all_messages.append(content)
                        
                        avg_length = sum(len(msg) for msg in all_messages) / len(all_messages) if all_messages else 0
                        
                        # Display stats
                        st.metric("Total Messages", message_count)
                        st.metric("Avg Message Length", f"{avg_length:.1f} chars")
                    except Exception as e:
                        st.error(f"Error analyzing persona: {str(e)}")
            
            # Input for the post
            st.subheader("LinkedIn Post to Comment On")
            post_text = st.text_area(
                "Paste LinkedIn post content here", 
                height=150,
                placeholder="Example: We just completed a major project migration to the cloud. The team pulled together amazingly to meet our deadline!"
            )
            
            # Post type classifier to help the model
            post_categories = ["Project Update", "Industry News", "Product Launch", "Company Achievement", "Thought Leadership", "Other"]
            post_category = st.selectbox(
                "Post category (helps generate better responses)", 
                options=post_categories
            )
            
            # Advanced settings
            with st.expander("Advanced Settings"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    k_similar = st.slider(
                        "Reference messages", 
                        min_value=3, 
                        max_value=20, 
                        value=7,
                        help="Number of similar historical messages to use as style reference"
                    )
                with col2:
                    max_tokens = st.slider(
                        "Max comment length", 
                        min_value=50, 
                        max_value=300, 
                        value=150,
                        help="Maximum length of the generated comment"
                    )
                with col3:
                    style_strength = st.select_slider(
                        "Style strength",
                        options=["Light", "Moderate", "Strong", "Very Strong"],
                        value="Strong",
                        help="How strongly to follow the persona's writing style"
                    )
            
            # Generate button
            if st.button("🚀 Generate Comment", type="primary", disabled=not post_text, use_container_width=True):
                if post_text:
                    try:
                        with st.spinner("AI is analyzing the persona style and generating a comment..."):
                            # Get persona path
                            persona_path = persona_manager.get_persona_path(selected_persona)
                            
                            # Initialize generator
                            generator = PersonaCommentGenerator(persona_path)
                            
                            # Modify prompt based on style strength and category
                            style_prompts = {
                                "Light": "Consider the following style but prioritize relevance",
                                "Moderate": "Use this style as a general guide",
                                "Strong": "Closely match this writing style",
                                "Very Strong": "Exactly replicate this writing style in your response"
                            }
                            
                            # Enhance the prompt with category and style info
                            enhanced_prompt = (
                                f"You are commenting on a {post_category} post. "
                                f"{style_prompts[style_strength]}. "
                                f"\n\nPost: {post_text}\n\nWrite an authentic comment:"
                            )
                            
                            # Generate comment
                            start_time = time.time()
                            comment = generator.generate_comment(
                                enhanced_prompt, 
                                k=k_similar, 
                                max_new_tokens=max_tokens
                            )
                            end_time = time.time()
                            
                            # Display results in a styled box
                            st.success(f"Generated in {end_time - start_time:.2f} seconds")
                            
                            result_container = st.container(border=True)
                            with result_container:
                                st.subheader("💬 Generated Comment")
                                st.info(comment)
                                
                                st.divider()
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.button("📋 Copy to Clipboard", use_container_width=True)
                                with col2:
                                    st.download_button(
                                        "💾 Save Comment",
                                        comment,
                                        file_name="persona_comment.txt",
                                        use_container_width=True
                                    )
                                
                                # Optional: Suggest improvements
                                if st.button("💡 Suggest Variations", use_container_width=True):
                                    st.write("Generating 2 alternative versions...")
                                    
                                    # Would implement alternative generation here in production
                                    st.info("Alternative 1: " + comment.replace(".", "!").replace("we", "our team"))
                                    st.info("Alternative 2: " + comment.replace("I", "We").replace(".", ". Really impressive!"))
                    except Exception as e:
                        st.error(f"Error generating comment: {str(e)}")
                        logger.error(f"Error in comment generation: {e}", exc_info=True)
    
    # Tab 2: Manage Personas
    with tab2:
        st.header("Manage Personas")
        
        # Display default persona info
        if has_default_persona:
            st.subheader("Default Persona")
            default_path = persona_manager.default_persona_path
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.success(f"✓ Richard's persona successfully loaded from: {default_path}")
                
                try:
                    with open(default_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    message_count = sum(len(month.get('messages', [])) for month in data)
                    st.write(f"Contains {message_count} messages for AI learning")
                except Exception as e:
                    st.warning(f"Could not analyze file: {str(e)}")
            
            with col2:
                st.info("Status: Active")
        
        st.divider()
        
        # Upload new persona
        st.subheader("Upload Additional Persona")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            uploaded_file = st.file_uploader(
                "Upload Persona JSON File", 
                type=["json"],
                help="Upload a JSON file containing the persona's historical messages"
            )
        with col2:
            persona_name = st.text_input(
                "Persona Name", 
                placeholder="e.g., CEO_David",
                help="Name to identify this persona"
            )
        
        if st.button("📤 Upload Persona", disabled=not (uploaded_file and persona_name), use_container_width=True):
            if uploaded_file and persona_name:
                try:
                    persona_manager.save_uploaded_persona(uploaded_file, persona_name)
                    st.success(f"Successfully uploaded persona: {persona_name}")
                    st.experimental_rerun()  # Refresh the page to show the new persona
                except Exception as e:
                    st.error(f"Error uploading persona: {str(e)}")
        
        # List existing personas
        st.subheader("Available Personas")
        
        # Add divider between persona types
        personas = persona_manager.list_personas()
        custom_personas = [p for p in personas if p != "Richard (Default)"]
        
        if not personas:
            st.info("No personas available. Upload one using the form above.")
        else:
            if custom_personas:
                st.write("**Custom Personas:**")
                for persona in custom_personas:
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.write(f"• {persona}")
                    with col2:
                        st.button("🗑️", key=f"del_{persona}", help="Delete this persona")
            else:
                st.write("No custom personas uploaded yet.")

def add_to_navigation():
    """
    Function to integrate this module into the main Streamlit app's navigation.
    Import and call this function from your main app.
    """
    return persona_comment_generator_page

# Direct execution for testing
if __name__ == "__main__":
    import sys
    def run_cli():
        persona_path = r"C:\Users\m.giritharan\linkedin_data_extractor\Richard Persona.json"
        generator = PersonaCommentGenerator(persona_path)
        test_post = "We just completed a TUPE consultation for 15 staff members across the IT service desk. Challenging but important for business continuity."
        print("Input Post:\n", test_post)
        print("\nGenerated CEO-style Comment:\n")
        comment = generator.generate_comment(test_post)
        print(comment)
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        run_cli()
    else:
        persona_comment_generator_page()
