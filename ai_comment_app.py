"""
Streamlit app for the AI-powered LinkedIn Comment Generator.
"""

import streamlit as st
from ai_comment_generator import AICommentGenerator

def main():
    """
    Streamlit app for the AI Comment Generator.
    """
    st.set_page_config(page_title="AI LinkedIn Comment Generator", page_icon="💬", layout="wide")
    
    st.title("AI LinkedIn Comment Generator")
    st.markdown("""
    This app generates professional, analytical comments for LinkedIn posts using AI.
    Enter a LinkedIn post or URL below to get started.
    """)
    
    # Initialize generator
    if 'generator' not in st.session_state:
        with st.spinner("Loading AI model... This may take a moment."):
            try:
                st.session_state.generator = AICommentGenerator()
                st.success("AI model loaded successfully!")
            except Exception as e:
                st.error(f"Error loading AI model: {str(e)}")
                st.stop()
    
    # Input area
    st.subheader("LinkedIn Post or URL")
    input_type = st.radio("Input type:", ["Post Content", "LinkedIn URL"])
    
    if input_type == "Post Content":
        input_text = st.text_area(
            "Enter LinkedIn post content:",
            height=150,
            placeholder="Paste the content of a LinkedIn post here..."
        )
    else:
        input_text = st.text_input(
            "Enter LinkedIn post URL:",
            placeholder="https://www.linkedin.com/posts/..."
        )
    
    # Generate button
    if st.button("Generate Comment", type="primary"):
        if not input_text:
            st.warning("Please enter a LinkedIn post or URL.")
        else:
            with st.spinner("Generating comment using AI..."):
                try:
                    comment = st.session_state.generator.generate_comment(input_text)
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
            "https://www.linkedin.com/posts/harshavarthini-sridhar-a529ba195_cybersecurity-nullbangalore-communitylearning-ugcPost-7325614377611448321-94sJ"
        ]
        
        for i, example in enumerate(examples):
            if st.button(f"Use Example {i+1}", key=f"example_{i}"):
                st.session_state.example_input = example
                if "linkedin.com" in example:
                    st.session_state.example_type = "LinkedIn URL"
                else:
                    st.session_state.example_type = "Post Content"
                st.rerun()
        
        if hasattr(st.session_state, 'example_input'):
            input_text = st.session_state.example_input
            input_type = st.session_state.example_type
            del st.session_state.example_input
            del st.session_state.example_type
    
    # Footer
    st.markdown("---")
    st.markdown("Powered by GPT-2 and trained on analytical communication styles.")

if __name__ == "__main__":
    main()