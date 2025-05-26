"""
Script to run the Streamlit app for Richard's LinkedIn Comment Generator.
"""

import streamlit as st
from linkedin_chatbot import LinkedInChatBot

def main():
    """
    Streamlit app for Richard's LinkedIn Comment Generator.
    """
    st.set_page_config(page_title="Richard's LinkedIn Comment Generator", page_icon="💬", layout="wide")
    
    st.title("Richard's LinkedIn Comment Generator")
    st.markdown("""
    This app generates professional comments for LinkedIn posts in Richard's distinctive style. 
    Enter a LinkedIn post below, and the AI will generate an analytical, well-structured comment 
    that matches Richard's communication style.
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
        ["facebook/opt-1.3b", "google/flan-t5-base", "gpt2", "EleutherAI/gpt-neo-125M"],
        index=0
    )
    
    # Initialize session state
    if 'chatbot' not in st.session_state or (
        st.session_state.get('embedding_model') != embedding_model or 
        st.session_state.get('llm_model') != llm_model
    ):
        with st.spinner("Loading models... This may take a few minutes."):
            try:
                st.session_state.chatbot = LinkedInChatBot(
                    data_json_path="Richard Persona.json",
                    embedding_model=embedding_model,
                    llm_model=llm_model
                )
                st.session_state.embedding_model = embedding_model
                st.session_state.llm_model = llm_model
                st.success("Models loaded successfully!")
            except Exception as e:
                error_msg = str(e)
                st.error(f"Error loading models: {error_msg}")
                st.markdown("""
                ### Troubleshooting Tips:
                1. Try selecting a different language model from the sidebar
                2. The 'gpt2' model is usually the most reliable option
                3. Make sure you have a stable internet connection
                4. Some models may require more memory than is available
                """)
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
                except Exception as e:
                    st.error(f"Error generating comment: {str(e)}")
    
    # Example posts
    with st.expander("Example Posts"):
        example_posts = [
            "Excited to announce that our company has just secured $10M in Series A funding! This investment will help us scale our operations and bring our innovative solution to more customers worldwide. #Startup #Funding #Innovation",
            "Just published a new article on the future of remote work. After interviewing 50+ business leaders, it's clear that hybrid models are here to stay. What's your experience with remote work been like? #RemoteWork #FutureOfWork",
            "Proud to share that our team has been recognized as one of the Top 10 Places to Work! This achievement reflects our commitment to creating a supportive, inclusive, and growth-oriented environment. #WorkplaceCulture #EmployeeExperience"
        ]
        
        for i, post in enumerate(example_posts):
            if st.button(f"Use Example {i+1}"):
                st.session_state.example_post = post
                st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("Powered by Hugging Face models and trained on Richard's communication style.")

if __name__ == "__main__":
    main()