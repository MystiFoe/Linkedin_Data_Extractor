# IMPORTANT: Richard's LinkedIn Comment Generator

## Model Access Issue

The original implementation attempted to use the `mistralai/Mistral-7B-Instruct-v0.2` model, which is a gated model requiring authentication. To address this issue, we've made the following changes:

1. Updated the default model to `facebook/opt-1.3b`, which is an open-source model
2. Added fallback to `gpt2` if the primary model fails to load
3. Created a simplified version (`simple_richard_bot.py`) that uses only `gpt2`

## Recommended Usage

For the most reliable experience, we recommend using the simplified version:

1. Double-click on `run_simple_richard_bot.bat`
2. OR run `streamlit run simple_richard_bot.py` from the command line

This version uses only the GPT-2 model, which is smaller and more likely to work without issues.

## Alternative Models

If you want to try different models, you can use the full version and select from:

- `facebook/opt-1.3b` (default)
- `google/flan-t5-base`
- `gpt2`
- `EleutherAI/gpt-neo-125M`

## Troubleshooting

If you encounter issues with model loading:

1. Try using the simplified version
2. Make sure you have a stable internet connection
3. Check that you have enough memory available
4. Try a smaller model like `gpt2`

## Files

- `simple_richard_bot.py`: Simplified version using only GPT-2
- `run_simple_richard_bot.bat`: Batch file to run the simplified version
- `linkedin_chatbot.py`: Full implementation with multiple model options
- `run_richard_bot.bat`: Batch file to run the full version

## Documentation

For complete documentation, see:
- `README_CHATBOT.md`: Detailed documentation
- `RICHARD_BOT_SUMMARY.md`: Summary of the implementation