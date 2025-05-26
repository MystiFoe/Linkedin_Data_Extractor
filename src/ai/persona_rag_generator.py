import os
import json
import logging
from typing import List, Dict, Any, Optional
import streamlit as st
import pandas as pd
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import chromadb
from chromadb.utils import embedding_functions
import hashlib
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Pre-load models at module level
@st.cache_resource(show_spinner="Loading AI models (this may take a few minutes)...")
def initialize_models(model_name="microsoft/Phi-4-mini-instruct", embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Initializing models on {device}")
    
    with st.spinner("Loading language model..."):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map=None
        )
    
    with st.spinner("Loading embedding model..."):
        embedding_model = SentenceTransformer(embedding_model, device=device)
    
    return tokenizer, model, embedding_model, device

# Initialize models at module level
tokenizer, model, embedding_model, device = initialize_models()

@st.cache_resource(show_spinner="Loading language model...")
def get_cached_model_and_tokenizer(llm_model_name):
    return tokenizer, model

@st.cache_resource(show_spinner="Loading embedding model...")
def get_cached_embedding_model(embedding_model_name, device):
    return embedding_model

@st.cache_resource(show_spinner="Building persona vector DB...")
def get_cached_vector_db(persona_data_path, embedding_model_name, device, collection_name):
    import chromadb
    import os
    import pandas as pd
    # Use file hash to invalidate cache if persona file changes
    with open(persona_data_path, 'rb') as f:
        persona_bytes = f.read()
    persona_hash = hashlib.md5(persona_bytes).hexdigest()
    # Use persistent ChromaDB storage
    db_dir = os.path.join(os.path.dirname(persona_data_path), "persona_chroma_db")
    chroma_client = chromadb.PersistentClient(path=db_dir)
    # Check if collection exists and has correct hash
    collection_names = [c.name for c in chroma_client.list_collections()]
    collection = None
    if collection_name in collection_names:
        collection = chroma_client.get_collection(collection_name)
        # Check if hash matches (store hash as a dummy doc with id 'persona_hash')
        try:
            hash_doc = collection.get(ids=["persona_hash"])
            if hash_doc and hash_doc['documents'] and hash_doc['documents'][0] == persona_hash:
                # Already embedded for this persona file
                messages = []  # Not needed for retrieval
                return collection, messages, persona_hash
        except Exception:
            pass
        # If hash doesn't match, delete and re-create
        chroma_client.delete_collection(collection_name)
        collection = None
    # (Re)create collection and embed
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=embedding_model_name,
        device=device
    )
    collection = chroma_client.create_collection(
        name=collection_name,
        embedding_function=sentence_transformer_ef
    )
    # Load (post, comment) pairs from Excel
    with st.spinner("Loading data from Excel..."):
        df = pd.read_excel(persona_data_path, engine="openpyxl")
        logger.info(f"Loaded {len(df)} rows from Excel")
    
    # Print column names for debugging
    logger.info(f"column names : Excel file : {persona_data_path}")
    logger.info(f"Columns : {' '.join(df.columns)}")
    
    # For the specific file skellett_comments_with_post_content.xlsx
    if "postContent" in df.columns and "Comment" in df.columns:
        post_col = "postContent"
        comment_col = "Comment"
        logger.info(f"Using columns: post='{post_col}', comment='{comment_col}'")
    else:
        # Fallback to generic column detection
        post_col = None
        comment_col = None
        for col in df.columns:
            if col.lower() == 'post' or col.lower() == 'postcontent' or col.lower() == 'content':
                post_col = col
            if col.lower() == 'comment' or col.lower() == 'comments':
                comment_col = col
        
        if not post_col or not comment_col:
            raise ValueError(f"Excel file must contain post and comment columns. Found columns: {', '.join(df.columns)}")
    # Build documents as: 'POST: ...\nCOMMENT: ...'
    docs = [f"POST: {row[post_col]}\nCOMMENT: {row[comment_col]}" for _, row in df.iterrows() if pd.notnull(row[post_col]) and pd.notnull(row[comment_col])]
    # Limit for dev speed
    max_docs = 500
    if len(docs) > max_docs:
        docs = docs[:max_docs]
    
    # Build documents and add to collection with progress bar
    with st.spinner("Processing and embedding documents..."):
        max_batch_size = 100  # Smaller batch size for more frequent updates
        total_docs = len(docs)
        progress_bar = st.progress(0)
        
        for i in range(0, total_docs, max_batch_size):
            end_idx = min(i + max_batch_size, total_docs)
            batch_docs = docs[i:end_idx]
            batch_ids = [f"pair_{j}" for j in range(i, end_idx)]
            
            # Update progress
            progress = float(i) / total_docs
            progress_bar.progress(progress)
            
            collection.add(documents=batch_docs, ids=batch_ids)
        
        # Final progress update
        progress_bar.progress(1.0)
        st.success(f"Successfully processed {total_docs} documents!")
    
    # Store the hash as a dummy doc
    collection.add(documents=[persona_hash], ids=["persona_hash"])
    return collection, docs, persona_hash

class PersonaRAGGenerator:
    """
    A class for generating comments based on user personas using RAG (Retrieval Augmented Generation)
    with the Microsoft Phi-4-mini-instruct model.
    """
    
    def __init__(
        self,
        persona_data_path: str,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model_name: str = "microsoft/Phi-4-mini-instruct",
        collection_name: str = "persona_pairs",
        device: str = None,
    ):
        """
        Initialize the PersonaRAGGenerator.
        
        Args:
            persona_data_path: Path to the JSON file containing the user's conversation history
            embedding_model_name: Name of the embedding model to use
            llm_model_name: Name of the language model to use for generation
            collection_name: Name of the ChromaDB collection to store embeddings
            device: Device to run the models on ('cuda', 'cpu', etc.)
        """
        self.persona_data_path = persona_data_path
        self.embedding_model_name = embedding_model_name
        self.llm_model_name = llm_model_name
        self.collection_name = collection_name
        self.device = device if device else "cuda" if torch.cuda.is_available() else "cpu"
        
        # Use globally initialized models
        self.tokenizer = tokenizer
        self.model = model
        self.embedding_model = embedding_model
        
        # Use cached vector DB (embeddings)
        with st.spinner("Initializing vector database..."):
            self.collection, self.docs, self.persona_hash = get_cached_vector_db(
                self.persona_data_path, self.embedding_model_name, self.device, self.collection_name
            )
        
        self.persona_name = os.path.basename(persona_data_path).replace('.xlsx', '')
    
    def _load_messages(self, json_path: str, max_messages: int = 500) -> List[str]:
        """
        Load messages from a JSON file containing conversation history.
        
        Args:
            json_path: Path to the JSON file
            max_messages: Maximum number of messages to load (default: 5000)
            
        Returns:
            List of message strings
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Extract messages from the conversation
            messages = []
            
            # Handle different JSON structures
            if isinstance(data, list):
                # If data is a list of messages
                for item in data:
                    if isinstance(item, dict) and 'content' in item:
                        messages.append(item['content'])
                    elif isinstance(item, str):
                        messages.append(item)
            elif isinstance(data, dict):
                # Special case for Richard Persona.json format
                for month, conversations in data.items():
                    if isinstance(conversations, list):
                        for conversation in conversations:
                            if 'messages' in conversation and isinstance(conversation['messages'], list):
                                for msg in conversation['messages']:
                                    if isinstance(msg, dict) and 'text' in msg and msg.get('author') == 'ChatGPT':
                                        messages.append(msg['text'])
                
                # If data is a dictionary with a messages field
                if 'messages' in data and isinstance(data['messages'], list):
                    for msg in data['messages']:
                        if isinstance(msg, dict):
                            if 'content' in msg:
                                messages.append(msg['content'])
                            elif 'text' in msg and msg.get('author', msg.get('role')) == 'ChatGPT':
                                messages.append(msg['text'])
                
                # If data contains conversation history in another format
                elif 'history' in data and isinstance(data['history'], list):
                    for entry in data['history']:
                        if isinstance(entry, dict) and 'message' in entry:
                            messages.append(entry['message'])
            
            # Limit the number of messages if there are too many
            if len(messages) > max_messages:
                logger.warning(f"Too many messages ({len(messages)}), limiting to {max_messages}")
                # Keep a mix of messages from different parts of the list for better diversity
                step = len(messages) // max_messages
                if step < 1:
                    step = 1
                
                # Take every 'step' message to get a representative sample
                limited_messages = messages[::step]
                
                # If we still have too many, just take the first max_messages
                if len(limited_messages) > max_messages:
                    limited_messages = limited_messages[:max_messages]
                
                messages = limited_messages
            
            logger.info(f"Loaded {len(messages)} messages from {json_path}")
            return messages
        except Exception as e:
            logger.error(f"Error loading messages from {json_path}: {e}")
            return []
    
    def _create_vector_db(self):
        """Create a vector database from the persona messages."""
        try:
            # Delete collection if it exists
            try:
                self.chroma_client.delete_collection(self.collection_name)
            except:
                pass
            
            # Create a new collection
            sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_model_name,
                device=self.device
            )
            
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                embedding_function=sentence_transformer_ef
            )
            
            # Add documents to the collection in batches to avoid exceeding max batch size
            if self.messages:
                # ChromaDB max batch size is 5461
                max_batch_size = 5461
                total_messages = len(self.messages)
                
                for i in range(0, total_messages, max_batch_size):
                    end_idx = min(i + max_batch_size, total_messages)
                    batch_messages = self.messages[i:end_idx]
                    batch_ids = [f"msg_{j}" for j in range(i, end_idx)]
                    
                    logger.info(f"Adding batch {i//max_batch_size + 1}/{(total_messages-1)//max_batch_size + 1} to vector database ({len(batch_messages)} messages)")
                    
                    self.collection.add(
                        documents=batch_messages,
                        ids=batch_ids
                    )
                
                logger.info(f"Created vector database with {total_messages} entries")
            else:
                logger.warning("No messages to add to vector database")
                
        except Exception as e:
            logger.error(f"Error creating vector database: {e}")
            raise
    
    def generate_comment(self, post_text: str, author_name: str, k: int = 1, max_new_tokens: int = 80) -> str:
        """
        Generate a comment for a LinkedIn post based on the persona's real (post, comment) pairs.
        
        Args:
            post_text: The text of the LinkedIn post
            author_name: The name of the post author
            k: Number of similar messages to retrieve (reduced default from 5 to 3)
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated comment
        """
        try:
            logger.info(f"Starting comment generation for post by {author_name} (k={k}, max_new_tokens={max_new_tokens})")
            start_time = time.time()
            # Query the vector database for similar (post, comment) pairs
            query_results = self.collection.query(
                query_texts=[post_text],
                n_results=k
            )
            similar_pairs = query_results['documents'][0]
            # Extract only the COMMENT part from each retrieved doc
            examples = []
            for doc in similar_pairs:
                # Expect format: 'POST: ...\nCOMMENT: ...'
                if '\nCOMMENT:' in doc:
                    comment_part = doc.split('\nCOMMENT:', 1)[1].strip()
                    # Optionally, also show the post for context
                    post_part = doc.split('\nCOMMENT:', 1)[0].replace('POST:', '').strip()
                    examples.append(f"Post: {post_part}\nComment: {comment_part}")
                else:
                    examples.append(doc)
            # Limit the length of each example
            truncated_examples = []
            for ex in examples:
                words = ex.split()
                if len(words) > 200:
                    truncated_ex = " ".join(words[:200]) + "..."
                else:
                    truncated_ex = ex
                truncated_examples.append(truncated_ex)
            # Join examples
            examples_str = "\n\n---\n\n".join(truncated_examples)
            # Prompt
            prompt = f"""You are Richard, an HR and business management expert. Write a LinkedIn comment in Richard's style.\n\nHere are real examples of Richard's comments on LinkedIn posts:\n\n{examples_str}\n\nWrite a comment on this new LinkedIn post by {author_name}:\n\"{post_text}\"\n\nYour comment should be professional, insightful, concise (100-150 words), include 1-2 hashtags, and match Richard's style.\nWrite ONLY the comment:"""

            logger.info(f"Prompt length: {len(prompt)} characters, {len(prompt.split())} words")
            if len(prompt) > 4000:
                logger.warning("Prompt is very long; generation may be slow.")
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"]
            
            # If the input is still too large, further reduce the examples
            if input_ids.size(1) > 5000:  # A conservative threshold
                logger.warning(f"Input size {input_ids.size(1)} is large, reducing examples")
                # Use fewer examples
                truncated_examples = truncated_examples[:1]  # Just use the most relevant example
                examples_str = "\n\n---\n\n".join(truncated_examples)
                
                # Even more concise prompt
                prompt = f"""You are Richard, an HR expert. Write a LinkedIn comment in Richard's style.

Example of Richard's comment:

{examples_str}

Comment on this post by {author_name}:
"{post_text}"

Be professional, concise (100 words), include a hashtag, match Richard's style.
Write ONLY the comment:"""
                
                inputs = self.tokenizer(prompt, return_tensors="pt")
            
            if self.device != "cpu":
                inputs = inputs.to(self.device)
            
            try:
                logger.info("Calling model.generate()...")
                gen_start = time.time()
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        repetition_penalty=1.2
                    )
                gen_end = time.time()
                logger.info(f"Model generation took {gen_end - gen_start:.2f} seconds.")
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            except ValueError as ve:
                if "batch size" in str(ve).lower():
                    logger.error(f"Batch size error: {ve}. Trying with minimal prompt.")
                    # Final fallback with minimal prompt
                    minimal_prompt = f"""Write a professional LinkedIn comment on this post by {author_name}: "{post_text}". 
Be concise (under 100 words) and include a relevant hashtag."""
                    
                    minimal_inputs = self.tokenizer(minimal_prompt, return_tensors="pt")
                    if self.device != "cpu":
                        minimal_inputs = minimal_inputs.to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **minimal_inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9
                        )
                    
                    generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    # Extract just the comment part (after the prompt)
                    comment = generated_text[len(minimal_prompt):].strip()
                    return comment
                else:
                    raise
            except Exception as e:
                logger.error(f"Error during model generation: {e}")
                return f"I couldn't generate a comment at this time. Please try again later. Error: {str(e)}"
            
            # Extract just the comment part (after the prompt)
            comment = generated_text[len(prompt):].strip()
            
            # Clean up the comment to make it more suitable for LinkedIn
            # Remove any potential artifacts or formatting issues
            comment = comment.replace("Comment:", "").strip()
            
            # Ensure the comment isn't too long for LinkedIn
            if len(comment.split()) > 200:
                # Truncate to approximately 150-200 words
                comment = " ".join(comment.split()[:180]) + "..."
            
            total_time = time.time() - start_time
            logger.info(f"Total comment generation time: {total_time:.2f} seconds.")
            if total_time > 60:
                return "[Warning: Comment generation took too long. Try reducing the post length or number of examples.]\n" + comment
            return comment
            
        except Exception as e:
            logger.error(f"Error generating comment: {e}")
            return f"Error generating comment: {str(e)}"


class PersonaRAGManager:
    """
    A class for managing persona models and their storage.
    """
    
    def __init__(self, base_dir: str = "persona_models"):
        """
        Initialize the PersonaRAGManager.
        
        Args:
            base_dir: Base directory for storing persona models
        """
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        
    def list_personas(self) -> List[str]:
        """
        List all available personas.
        
        Returns:
            List of persona names
        """
        try:
            persona_files = [f for f in os.listdir(self.base_dir) if f.endswith('.json')]
            return [f.replace('.json', '') for f in persona_files]
        except Exception as e:
            logger.error(f"Error listing personas: {e}")
            return []
    
    def get_persona_path(self, persona_name: str) -> str:
        """
        Get the path to a persona file.
        
        Args:
            persona_name: Name of the persona
            
        Returns:
            Path to the persona file
        """
        return os.path.join(self.base_dir, f"{persona_name}.json")
    
    def save_uploaded_persona(self, uploaded_file, persona_name: str) -> str:
        """
        Save an uploaded persona file.
        
        Args:
            uploaded_file: Uploaded file object
            persona_name: Name to save the persona as
            
        Returns:
            Path to the saved persona file
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(self.base_dir, exist_ok=True)
            
            # Save the file
            file_path = self.get_persona_path(persona_name)
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
                
            logger.info(f"Saved persona {persona_name} to {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Error saving persona: {e}")
            raise


def persona_rag_generator_page():
    """
    Streamlit page for the Persona RAG Generator.
    """
    st.title("Richard Persona LinkedIn Comment Generator")
    st.write("Generate professional LinkedIn comments in Richard's style using RAG and the Phi-4 model")
    
    # Initialize the persona manager
    persona_manager = PersonaRAGManager()
    
    # Ensure Richard persona is available
    richard_persona_path = "c:/Users/m.giritharan/linkedin_data_extractor/skellett_comments_with_post_content.xlsx"
    if not os.path.exists(richard_persona_path):
        st.error("skellett_comments_with_post_content.xlsx file not found. Please ensure the file exists at the correct location.")
        return
    
    # Create a styled container for the LinkedIn post input
    st.markdown("### LinkedIn Post Information")
    with st.container():
        st.markdown("""
        <style>
        .linkedin-container {
            background-color: #f3f6f8;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Input for LinkedIn post
        post_text = st.text_area("Enter the LinkedIn post text", 
                                height=150,
                                placeholder="Paste the LinkedIn post content here...")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            author_name = st.text_input("Post Author Name", 
                                      placeholder="Enter the name of the post author")
        
        with col2:
            # Advanced settings in a more compact form
            with st.expander("Advanced Settings"):
                k_value = st.slider("Number of examples to use", 3, 10, 5)
                max_tokens = st.slider("Maximum comment length", 100, 300, 150)
    
    # Generate comment button with styling
    generate_btn = st.button("Generate Comment in Richard's Style", type="primary", use_container_width=True)
    
    if generate_btn and post_text and author_name:
        import datetime
        gen_start_time = time.time()
        st.info("⏳ If generation takes more than 30 seconds, try reducing the post length or number of examples.")
        with st.spinner("Analyzing Richard's communication style and generating comment..."):
            try:
                logger.info(f"[UI] Starting comment generation at {datetime.datetime.now().isoformat()} for author '{author_name}'")
                # Initialize the generator with Richard's persona
                generator = PersonaRAGGenerator(richard_persona_path)
                # Generate the comment
                comment = generator.generate_comment(
                    post_text=post_text,
                    author_name=author_name,
                    k=k_value,
                    max_new_tokens=max_tokens
                )
                gen_end_time = time.time()
                elapsed = gen_end_time - gen_start_time
                logger.info(f"[UI] Finished comment generation in {elapsed:.2f} seconds.")
                if elapsed > 30:
                    st.warning(f"⚠️ Comment generation took {elapsed:.1f} seconds. Consider reducing the post length or number of examples for faster results.")
                # Display the result in a styled container
                st.markdown("### Generated Comment")
                # Process the comment to replace newlines with HTML breaks
                formatted_comment = comment.replace('\n', '<br>')
                st.markdown(f"""
                <div style="background-color: #e6f2ff; border-left: 5px solid #0077b5; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
                {formatted_comment}
                </div>
                """, unsafe_allow_html=True)
                # Copy functionality
                st.text_area("Copy Comment", comment, height=100)
                # Add some metrics about the comment
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Word Count", len(comment.split()))
                with col2:
                    st.metric("Character Count", len(comment))
                with col3:
                    # Count hashtags
                    hashtag_count = comment.count('#')
                    st.metric("Hashtags", hashtag_count)
            except Exception as e:
                logger.error(f"[UI] Error generating comment: {e}", exc_info=True)
                st.error(f"Error generating comment: {e}")
    
    # Tips for effective LinkedIn comments
    with st.expander("Tips for Effective LinkedIn Comments"):
        st.markdown("""
        ### Best Practices for LinkedIn Engagement
        
        1. **Add Value**: Provide insights, experiences, or perspectives that extend the conversation
        2. **Be Authentic**: Write in a genuine voice that reflects Richard's professional tone
        3. **Keep it Concise**: LinkedIn comments perform best when they're focused and to the point
        4. **Ask Questions**: End with a thoughtful question to encourage further engagement
        5. **Use Formatting**: Break up longer comments with line breaks for readability
        6. **Include Relevant Hashtags**: Add 1-2 targeted hashtags to increase visibility
        
        The generated comments follow Richard's analytical and strategic communication style while adhering to these best practices.
        """)


def add_to_navigation():
    """
    Add the Richard Persona Comment Generator page to the navigation.
    This function is called from the main app.py file.
    """
    return {
        "Richard Comment Generator": persona_rag_generator_page
    }


if __name__ == "__main__":
    # For testing the module independently
    import streamlit as st
    
    persona_rag_generator_page()