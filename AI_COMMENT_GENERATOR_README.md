# AI LinkedIn Comment Generator

This is an AI-powered comment generator for LinkedIn posts that:

1. Uses the GPT-2 language model
2. Handles both LinkedIn post content and URLs
3. Generates professional, analytical comments
4. Is trained on examples from Richard's communication style

## How to Use

### Option 1: Streamlit App (Recommended)

1. Double-click on `run_ai_comment_generator.bat`
2. Enter a LinkedIn post or URL in the input field
3. Click "Generate Comment"
4. Copy the generated comment

### Option 2: Command Line (Quick)

1. Double-click on `generate_comment.bat`
2. Enter a LinkedIn post or URL when prompted
3. The comment will be generated and displayed

### Option 3: Python Script

```bash
python ai_comment_generator.py --post "Your LinkedIn post or URL here"
```

Or to read from a file:

```bash
python ai_comment_generator.py --file path/to/post.txt
```

## Features

- **AI-Powered**: Uses GPT-2 for natural language generation
- **URL Handling**: Properly processes LinkedIn URLs
- **Example-Based**: Learns from Richard's communication style
- **User-Friendly**: Simple interface with copy functionality
- **Reliable**: Includes fallback mechanisms if generation fails

## Files

- `ai_comment_generator.py`: Core implementation of the AI generator
- `ai_comment_app.py`: Streamlit app for the generator
- `run_ai_comment_generator.bat`: Batch file to run the Streamlit app
- `generate_comment.bat`: Command-line interface for quick generation

## Troubleshooting

If you encounter issues:

1. Make sure you have an internet connection for the initial model download
2. Ensure you have enough memory available (GPT-2 requires ~500MB)
3. Try restarting the application if it becomes unresponsive
4. For URLs, the generator will create a generic comment since it can't access the actual content