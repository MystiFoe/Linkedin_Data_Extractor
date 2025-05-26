import streamlit as st
from linkedin_chatbot import LinkedInChatBot

def main():
    """
    Standalone Streamlit app for the LinkedIn Comment Generator.
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
                except Exception as e:
                    st.error(f"Error generating comment: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("Powered by Hugging Face models and trained on professional LinkedIn interactions.")

if __name__ == "__main__":
    main()