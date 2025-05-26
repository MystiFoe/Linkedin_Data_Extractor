"""
Test script for the LinkedIn chatbot.
"""

from linkedin_chatbot import LinkedInChatBot

def main():
    """
    Test the LinkedIn chatbot with a sample post.
    """
    print("Initializing LinkedIn ChatBot...")
    chatbot = LinkedInChatBot()
    
    test_post = """
    Excited to announce that our company has just secured $10M in Series A funding! 
    This investment will help us scale our operations and bring our innovative solution 
    to more customers worldwide. #Startup #Funding #Innovation
    """
    
    print("\nGenerating comment for test post...")
    comment = chatbot.generate_comment(test_post)
    
    print("\n--- Generated Comment ---")
    print(comment)
    print("------------------------\n")

if __name__ == "__main__":
    main()