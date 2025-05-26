import streamlit as st
import sys
import os
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from linkedin_chatbot import LinkedInChatBot
from src.logger import app_logger

def linkedin_comment_page():
    """
    Streamlit page for the LinkedIn Comment Generator.
    """
    st.title("Richard's LinkedIn Comment Generator")
    st.markdown("""
    Generate professional comments for LinkedIn posts in Richard's distinctive style. 
    The AI is trained on Richard's previous responses and will create 
    analytical, well-structured comments that match his communication style.
    """)
    
    # Sidebar for model selection
    with st.sidebar:
        st.header("Model Settings")
        embedding_model = st.selectbox(
            "Embedding Model",
            ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2"],
            index=0
        )
        
        llm_model = st.selectbox(
            "Language Model",
            ["facebook/opt-1.3b", "google/flan-t5-base", "gpt2", "EleutherAI/gpt-neo-125M"],
            index=0
        )
        
        st.markdown("---")
        st.markdown("**Note:** Changing models will reload the AI, which may take a few minutes.")
        
        if st.button("Reload Models"):
            st.session_state.pop('linkedin_chatbot', None)
            st.success("Models will be reloaded!")
    
    # Initialize session state for the chatbot
    if 'linkedin_chatbot' not in st.session_state or (
        st.session_state.get('embedding_model') != embedding_model or 
        st.session_state.get('llm_model') != llm_model
    ):
        with st.spinner("Loading models... This may take a few minutes."):
            try:
                # Get the absolute path to data.json
                data_json_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "model", "data.json"
                )
                
                st.session_state.linkedin_chatbot = LinkedInChatBot(
                    data_json_path=data_json_path,
                    embedding_model=embedding_model,
                    llm_model=llm_model
                )
                st.session_state.embedding_model = embedding_model
                st.session_state.llm_model = llm_model
                st.success("Models loaded successfully!")
            except Exception as e:
                error_msg = str(e)
                app_logger.error(f"Error loading models: {error_msg}")
                st.error(f"Error loading models: {error_msg}")
                st.markdown("""
                ### Troubleshooting Tips:
                1. Try selecting a different language model from the sidebar
                2. The 'gpt2' model is usually the most reliable option
                3. Make sure you have a stable internet connection
                4. Some models may require more memory than is available
                """)
                st.stop()
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("LinkedIn Post")
        post_text = st.text_area(
            "Enter the LinkedIn post you want to comment on:",
            height=300,
            placeholder="Paste the LinkedIn post here..."
        )
        
        generate_button = st.button("Generate Comment", type="primary")
        
        # Example posts
        with st.expander("Example Posts"):
            example_posts = [
                "Excited to announce that our company has just secured $10M in Series A funding! This investment will help us scale our operations and bring our innovative solution to more customers worldwide. #Startup #Funding #Innovation",
                "Just published a new article on the future of remote work. After interviewing 50+ business leaders, it's clear that hybrid models are here to stay. What's your experience with remote work been like? #RemoteWork #FutureOfWork",
                "Proud to share that our team has been recognized as one of the Top 10 Places to Work! This achievement reflects our commitment to creating a supportive, inclusive, and growth-oriented environment. #WorkplaceCulture #EmployeeExperience"
            ]
            
            for i, post in enumerate(example_posts):
                if st.button(f"Use Example {i+1}", key=f"example_{i}"):
                    st.session_state.example_post = post
                    st.rerun()
        
        # Use example post if selected
        if hasattr(st.session_state, 'example_post'):
            post_text = st.session_state.example_post
            # Clear the example post from session state
            del st.session_state.example_post
    
    with col2:
        st.subheader("Generated Comment")
        
        if generate_button and post_text:
            with st.spinner("Generating comment..."):
                try:
                    comment = st.session_state.linkedin_chatbot.generate_comment(post_text)
                    st.session_state.generated_comment = comment
                except Exception as e:
                    app_logger.error(f"Error generating comment: {str(e)}")
                    st.error(f"Error generating comment: {str(e)}")
        
        # Display generated comment if available
        if hasattr(st.session_state, 'generated_comment'):
            st.markdown(f"**Comment:**")
            st.markdown(st.session_state.generated_comment)
            
            # Copy button
            st.text_area(
                "Copy this comment:",
                value=st.session_state.generated_comment,
                height=200
            )
            
            if st.button("Clear Comment"):
                del st.session_state.generated_comment
                st.rerun()
        else:
            st.info("Your generated comment will appear here.")
    
    # Tips section
    st.markdown("---")
    with st.expander("Tips for Better Comments"):
        st.markdown("""
        ### Tips for Better LinkedIn Comments
        
        1. **Provide context in your post**: The more specific your post is, the more tailored the comment will be.
        2. **Include industry-specific terms**: This helps the AI generate more relevant comments.
        3. **Specify the tone**: If you want a specific tone (enthusiastic, thoughtful, analytical), include that in your post.
        4. **Try different models**: Different language models may generate different styles of comments.
        """)

def add_to_navigation():
    """
    Add the LinkedIn Comment Generator page to the navigation.
    """
    return {
        "name": "LinkedIn Comment Generator",
        "icon": "💬",
        "function": linkedin_comment_page
    }