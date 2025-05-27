import streamlit as st
import subprocess
import sys

def main():
    # Simple navigation bar
    st.markdown("""
    <nav style='background-color: #262730; padding: 10px 0; margin-bottom: 30px;'>
        <h2 style='color: #fff; text-align: center; margin: 0;'>AI Comment Generator (Blended Persona)</h2>
    </nav>
    """, unsafe_allow_html=True)
    
    st.header("Generate a LinkedIn Comment (Blended Persona)")

    post_content = st.text_area("Enter LinkedIn post content:", height=200)

    if st.button("Generate Comment"):
        if not post_content.strip():
            st.warning("Please enter post content.")
        else:
            with st.spinner("Generating comment via CLI..."):
                try:
                    result = subprocess.run([
                        sys.executable, "linkedin_chatbot_cli.py",
                        "--post", post_content,
                        "--embedding-model", "sentence-transformers/all-MiniLM-L6-v2",
                        "--llm-model", "facebook/opt-1.3b",
                        "--data-json", "Skellett.json"
                    ], capture_output=True, text=True, timeout=300)
                    output = result.stdout
                    error = result.stderr
                    if result.returncode == 0:
                        if "--- Generated Comment ---" in output:
                            comment = output.split("--- Generated Comment ---")[-1].split("------------------------")[0].strip()
                            st.success("Generated Comment:")
                            st.write(comment)
                            st.code(comment)
                        else:
                            st.error("Could not parse generated comment. Raw output:")
                            st.text(output)
                    else:
                        st.error(f"Error running CLI: {error or output}")
                except Exception as e:
                    st.error(f"Exception running CLI: {e}")

    st.sidebar.info("This app uses the CLI to generate unique, meaningful LinkedIn comments in a blended persona style.")

if __name__ == "__main__":
    main()
