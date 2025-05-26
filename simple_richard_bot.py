"""
Simplified version of Richard's LinkedIn Comment Generator using only GPT-2.
"""

import json
import logging
import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleRichardBot:
    """
    A simplified version of Richard's LinkedIn Comment Generator using only GPT-2.
    """
    
    def __init__(self, data_json_path="Richard Persona.json"):
        """
        Initialize the bot with Richard's persona data.
        """
        logger.info("Initializing SimpleRichardBot")
        
        # Load GPT-2 model
        try:
            self.generator = pipeline("text-generation", model="gpt2")
            logger.info("Successfully loaded GPT-2 model")
        except Exception as e:
            logger.error(f"Failed to load GPT-2 model: {e}")
            raise
        
        # Load embedding model
        try:
            self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            logger.info("Successfully loaded embedding model")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
        
        # Load Richard's persona data
        try:
            self.examples = self._load_examples(data_json_path)
            logger.info(f"Loaded {len(self.examples)} examples from Richard's persona data")
        except Exception as e:
            logger.error(f"Failed to load persona data: {e}")
            raise
    
    def _load_examples(self, json_path):
        """
        Load examples from Richard's persona data.
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        examples = []
        for month, entries in data.items():
            for entry in entries:
                for message in entry.get('messages', []):
                    if message.get('author') == 'ChatGPT':
                        content = message.get('text', '').strip()
                        if content:
                            # Extract a shorter snippet for use as an example
                            if len(content) > 300:
                                content = content[:300] + "..."
                            examples.append(content)
        
        # Return a subset of examples to avoid overwhelming the model
        return examples[:5]
    
    def generate_comment(self, post_text, max_length=250):
        """
        Generate a comment for a LinkedIn post in Richard's style.
        """
        logger.info(f"Generating comment for post: {post_text[:50]}...")
        
        # Create a prompt with Richard's examples
        prompt = (
            "You are Richard, a professional LinkedIn commenter with an analytical style. "
            "Write a comment on this LinkedIn post in your style.\n\n"
            "Examples of your previous comments:\n"
        )
        
        # Add examples
        for i, example in enumerate(self.examples):
            prompt += f"Example {i+1}:\n{example}\n\n"
        
        # Add the post
        prompt += f"LinkedIn Post:\n{post_text}\n\nYour comment:"
        
        # Generate comment
        try:
            result = self.generator(prompt, max_length=len(prompt.split()) + max_length, 
                                   do_sample=True, temperature=0.7)
            generated_text = result[0]['generated_text']
            
            # Extract the comment part
            if "Your comment:" in generated_text:
                comment = generated_text.split("Your comment:", 1)[1].strip()
            else:
                # Extract text after the prompt
                prompt_end = prompt.strip()[-20:]  # Last 20 chars of prompt for reliable splitting
                if prompt_end in generated_text:
                    comment = generated_text.split(prompt_end, 1)[1].strip()
                else:
                    comment = generated_text.replace(prompt, "").strip()
            
            logger.info(f"Successfully generated comment: {comment[:50]}...")
            return comment
        except Exception as e:
            logger.error(f"Failed to generate comment: {e}")
            raise

def main():
    """
    Streamlit app for the simplified Richard bot.
    """
    st.set_page_config(page_title="Richard's LinkedIn Comment Generator (Simple)", 
                      page_icon="💬", layout="wide")
    
    st.title("Richard's LinkedIn Comment Generator")
    st.markdown("""
    This simplified version generates LinkedIn comments in Richard's style using GPT-2.
    Enter a LinkedIn post below to get started.
    """)
    
    # Initialize bot
    if 'bot' not in st.session_state:
        with st.spinner("Loading models... This may take a moment."):
            try:
                st.session_state.bot = SimpleRichardBot()
                st.success("Models loaded successfully!")
            except Exception as e:
                st.error(f"Error loading models: {str(e)}")
                st.stop()
    
    # Input area
    post_text = st.text_area("Enter LinkedIn Post", height=200)
    
    # Generate button
    if st.button("Generate Comment"):
        if not post_text:
            st.warning("Please enter a LinkedIn post.")
        else:
            with st.spinner("Generating comment..."):
                try:
                    comment = st.session_state.bot.generate_comment(post_text)
                    st.subheader("Generated Comment:")
                    st.write(comment)
                    
                    # Copy area
                    st.text_area("Copy this comment:", value=comment, height=150)
                except Exception as e:
                    st.error(f"Error generating comment: {str(e)}")
    
    # Example posts
    with st.expander("Example Posts"):
        examples = [
            "Excited to announce that our company has just secured $10M in Series A funding! This investment will help us scale our operations and bring our innovative solution to more customers worldwide. #Startup #Funding #Innovation",
            "Just published a new article on the future of remote work. After interviewing 50+ business leaders, it's clear that hybrid models are here to stay. What's your experience with remote work been like? #RemoteWork #FutureOfWork",
            "Proud to share that our team has been recognized as one of the Top 10 Places to Work! This achievement reflects our commitment to creating a supportive, inclusive, and growth-oriented environment. #WorkplaceCulture #EmployeeExperience"
        ]
        
        for i, example in enumerate(examples):
            if st.button(f"Use Example {i+1}", key=f"example_{i}"):
                st.session_state.example_post = example
                st.rerun()
        
        if hasattr(st.session_state, 'example_post'):
            post_text = st.session_state.example_post
            del st.session_state.example_post
    
    # Footer
    st.markdown("---")
    st.markdown("Powered by GPT-2 and trained on Richard's communication style.")

if __name__ == "__main__":
    main()