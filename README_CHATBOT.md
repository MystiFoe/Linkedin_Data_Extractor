# Richard's LinkedIn Comment Generator

This chatbot uses a base model from Hugging Face and leverages data from `Richard Persona.json` to generate professional comments for LinkedIn posts in Richard's distinctive analytical style.

## Features

- Uses sentence embeddings to find similar content in the training data
- Generates contextually relevant comments based on the LinkedIn post content
- Integrates with the existing LinkedIn Data Extractor application
- Can be run as a standalone application

## Installation

1. Install the required dependencies:

```bash
pip install transformers sentence-transformers faiss-cpu torch streamlit
```

2. Make sure you have the `Richard Persona.json` file in the project directory.

## Usage

### Option 1: Run as part of the LinkedIn Data Extractor

```bash
python app.py
```

Then select "LinkedIn Comment Generator" from the navigation menu.

### Option 2: Run the standalone comment generator

```bash
python run_comment_generator.py
```

### Option 3: Use the command-line interface

```bash
python linkedin_chatbot_cli.py --post "Your LinkedIn post text here"
```

Or to read from a file:

```bash
python linkedin_chatbot_cli.py --file path/to/post.txt
```

## How It Works

1. The chatbot loads the `Richard Persona.json` file and extracts Richard's messages for training.
2. It uses a sentence transformer model to create embeddings for these messages.
3. When a LinkedIn post is provided, it finds similar content in Richard's previous responses.
4. It then uses a language model to generate a comment in Richard's distinctive style based on the post and similar content.

## Customization

You can customize the models used by the chatbot:

- Embedding Model: Controls how the chatbot understands the content
  - Default: `sentence-transformers/all-MiniLM-L6-v2`
  - Alternative: `sentence-transformers/all-mpnet-base-v2`

- Language Model: Controls the generation of comments
  - Default: `facebook/opt-1.3b`
  - Alternatives: `google/flan-t5-base`, `gpt2`, `EleutherAI/gpt-neo-125M`

## Files

- `linkedin_chatbot.py`: Main chatbot implementation
- `linkedin_chatbot_cli.py`: Command-line interface
- `run_comment_generator.py`: Standalone Streamlit application
- `src/ui/linkedin_comment_page.py`: Integration with LinkedIn Data Extractor

## Tips for Better Results

1. Provide detailed LinkedIn posts for more contextual comments
2. Include industry-specific terms in your posts
3. Try different models for different styles of comments
4. The more diverse your training data, the better the comments will be