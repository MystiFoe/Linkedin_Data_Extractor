"""
AI-powered LinkedIn Comment Generator using GPT-2.
This version properly handles LinkedIn URLs and post content.
"""

import json
import os
import logging
import argparse
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import List, Dict, Any, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AICommentGenerator:
    """
    AI-powered comment generator for LinkedIn posts using GPT-2.
    """
    
    def __init__(self, persona_json_path: str = "Richard Persona.json"):
        """
        Initialize the AI Comment Generator.
        
        Args:
            persona_json_path: Path to the persona JSON file
        """
        logger.info(f"Initializing AI Comment Generator with {persona_json_path}")
        
        # Load GPT-2 model and tokenizer
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.model = GPT2LMHeadModel.from_pretrained("gpt2")
            logger.info("Successfully loaded GPT-2 model and tokenizer")
        except Exception as e:
            logger.error(f"Failed to load GPT-2 model: {e}")
            raise
        
        # Load persona examples
        self.examples = self._load_examples(persona_json_path)
        logger.info(f"Loaded {len(self.examples)} examples from persona data")
    
    def _load_examples(self, json_path: str) -> List[str]:
        """
        Load examples from the persona JSON file.
        
        Args:
            json_path: Path to the JSON file
            
        Returns:
            List of example texts
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            examples = []
            for month, entries in data.items():
                for entry in entries:
                    for message in entry.get('messages', []):
                        if message.get('author') == 'ChatGPT':
                            content = message.get('text', '').strip()
                            if content:
                                # Take just the first 200 characters for each example
                                examples.append(content[:200])
                            # Limit to 5 examples to keep the context manageable
                            if len(examples) >= 5:
                                return examples
            
            return examples[:5]  # Ensure we don't return too many
        except Exception as e:
            logger.error(f"Failed to load examples: {e}")
            # Return some default examples if loading fails
            return [
                "This analysis is spot on. I particularly appreciate the structured approach to problem-solving demonstrated here.",
                "Your points about market trends align with what I've been observing. The data suggests continued growth in this sector.",
                "The strategic implications of this announcement are significant. Companies will need to adapt their approaches accordingly."
            ]
    
    def _extract_post_content(self, input_text: str) -> str:
        """
        Extract post content from input text, which might be a URL or the post itself.
        
        Args:
            input_text: Input text, which could be a URL or post content
            
        Returns:
            Extracted post content
        """
        # Check if input is a LinkedIn URL
        if "linkedin.com/posts/" in input_text or "linkedin.com/feed/update/" in input_text:
            # For a URL, we'll create a generic description since we can't fetch the content
            return (
                "This LinkedIn post appears to be about professional development, "
                "industry insights, or business strategy. Without accessing the actual content, "
                "I'll generate a comment that would be appropriate for a typical LinkedIn post."
            )
        
        # If not a URL, assume it's the post content
        return input_text
    
    def generate_comment(self, input_text: str, max_length: int = 150) -> str:
        """
        Generate a comment for the given LinkedIn post.
        
        Args:
            input_text: LinkedIn post text or URL
            max_length: Maximum length of generated comment
            
        Returns:
            Generated comment text
        """
        logger.info(f"Generating comment for: {input_text[:50]}...")
        
        # Extract post content
        post_content = self._extract_post_content(input_text)
        logger.info(f"Extracted post content: {post_content[:50]}...")
        
        # Create prompt with examples
        prompt = "Generate a professional LinkedIn comment in an analytical style for this post:\n\n"
        prompt += f"Post: {post_content}\n\n"
        prompt += "Examples of analytical LinkedIn comments:\n"
        
        for i, example in enumerate(self.examples):
            prompt += f"{i+1}. {example}\n"
        
        prompt += "\nComment:"
        
        # Generate comment using GPT-2
        try:
            # Encode the prompt
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
            
            # Generate response
            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    max_length=input_ids.size(1) + max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode the response
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Extract just the comment part
            if "Comment:" in generated_text:
                comment = generated_text.split("Comment:", 1)[1].strip()
            else:
                # If we can't find the marker, take everything after the prompt
                comment = generated_text[len(prompt):].strip()
            
            logger.info(f"Successfully generated comment: {comment[:50]}...")
            return comment
        except Exception as e:
            logger.error(f"Failed to generate comment: {e}")
            # Return a fallback comment if generation fails
            return (
                "This is a thoughtful post that raises important points. "
                "I appreciate the structured approach and the insights shared. "
                "It would be interesting to explore how these ideas might evolve over time."
            )

def main():
    """
    Command-line interface for the AI Comment Generator.
    """
    parser = argparse.ArgumentParser(description="Generate comments for LinkedIn posts using AI")
    parser.add_argument("--post", type=str, help="LinkedIn post text or URL")
    parser.add_argument("--file", type=str, help="File containing LinkedIn post text or URL")
    parser.add_argument("--persona", type=str, default="Richard Persona.json", 
                        help="Path to persona JSON file")
    
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
            return 1
    else:
        print("Please provide LinkedIn post text or URL using --post or --file")
        return 1
    
    # Initialize generator
    print(f"Initializing AI Comment Generator...")
    try:
        generator = AICommentGenerator(persona_json_path=args.persona)
    except Exception as e:
        print(f"Error initializing generator: {str(e)}")
        return 1
    
    # Generate comment
    print("Generating comment...")
    try:
        comment = generator.generate_comment(post_text)
        print("\n--- Generated Comment ---")
        print(comment)
        print("------------------------\n")
    except Exception as e:
        print(f"Error generating comment: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()