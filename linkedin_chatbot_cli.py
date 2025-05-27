import argparse
import sys
from linkedin_chatbot_new import LinkedInChatBot

def main():
    """
    Command-line interface for the LinkedIn ChatBot.
    """
    parser = argparse.ArgumentParser(description="Generate comments for LinkedIn posts in Skellett's style")
    parser.add_argument("--post", type=str, help="LinkedIn post text")
    parser.add_argument("--file", type=str, help="File containing LinkedIn post text")
    parser.add_argument("--embedding-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", 
                        help="Embedding model to use")
    parser.add_argument("--llm-model", type=str, default="facebook/opt-1.3b", 
                        help="Language model to use")
    parser.add_argument("--data-json", type=str, default="Skellett.json", 
                        help="Path to persona data file")
    
    args = parser.parse_args()
    
    # Get post text from either command line argument or file
    post_text = None
    if args.post:
        post_text = args.post
    elif args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                post_text = f.read()
        except Exception as e:
            print(f"Error reading file: {str(e)}")
            sys.exit(1)
    else:
        print("Please provide LinkedIn post text using --post or --file")
        sys.exit(1)
    
    # Use Skellett.json as the data source
    data_json = args.data_json if args.data_json else "Skellett.json"
    print(f"Using data source: {data_json}")
    
    # Initialize chatbot
    print(f"Initializing LinkedIn ChatBot with {args.embedding_model} and {args.llm_model}...")
    try:
        chatbot = LinkedInChatBot(
            data_json_path=data_json,
            embedding_model=args.embedding_model,
            llm_model=args.llm_model
        )
    except Exception as e:
        print(f"Error initializing chatbot: {str(e)}")
        sys.exit(1)
    
    # Generate comment
    print("Generating comment...")
    try:
        comment = chatbot.generate_comment(post_text)
        print("\n--- Generated Comment ---")
        # Handle encoding issues by replacing problematic characters
        try:
            print(comment)
        except UnicodeEncodeError:
            # Replace problematic characters with '?' for display
            print(comment.encode('ascii', 'replace').decode('ascii'))
            print("\nNote: Some special characters were replaced due to encoding issues.")
        print("------------------------\n")
    except Exception as e:
        print(f"Error generating comment: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()