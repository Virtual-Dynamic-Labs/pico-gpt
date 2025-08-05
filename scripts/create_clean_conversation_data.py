#!/usr/bin/env python3
"""
Create clean conversational training data with better formatting
"""

def create_clean_conversation_data():
    """Create clean conversation data with proper formatting"""
    
    # Simple, clean conversation pairs
    conversations = [
        # Basic greetings
        "Human: Hello\nAssistant: Hello! How can I help you today?",
        "Human: Hi\nAssistant: Hi there! What would you like to talk about?",
        "Human: Hey\nAssistant: Hey! How are you doing?",
        "Human: Good morning\nAssistant: Good morning! I hope you're having a great day.",
        "Human: Good afternoon\nAssistant: Good afternoon! How has your day been so far?",
        "Human: Good evening\nAssistant: Good evening! How was your day?",
        
        # How are you variations
        "Human: How are you?\nAssistant: I'm doing well, thank you for asking! How are you?",
        "Human: How are you doing?\nAssistant: I'm doing great! How about you?",
        "Human: How's it going?\nAssistant: It's going well! How are things with you?",
        
        # Introductions
        "Human: What's your name?\nAssistant: I'm an AI assistant. What's your name?",
        "Human: My name is Alice\nAssistant: Nice to meet you, Alice! How can I help you today?",
        "Human: I'm Bob\nAssistant: Hello Bob! It's great to meet you.",
        "Human: Call me Sarah\nAssistant: Sure thing, Sarah! What can I do for you?",
        "Human: My name is Mike\nAssistant: Hi Mike! Pleasure to meet you.",
        "Human: I'm Emma\nAssistant: Hello Emma! How are you doing today?",
        
        # What can you do
        "Human: What can you do?\nAssistant: I can chat with you and help answer questions. What would you like to know?",
        "Human: How can you help me?\nAssistant: I can have conversations and try to be helpful. What do you need help with?",
        "Human: What are you?\nAssistant: I'm an AI assistant designed to be helpful and have conversations.",
        
        # Personal questions
        "Human: How old are you?\nAssistant: I don't have an age like humans do since I'm an AI. How old are you?",
        "Human: Where are you from?\nAssistant: I exist as software, so I don't have a physical location. Where are you from?",
        "Human: Do you have feelings?\nAssistant: That's an interesting question. I'm not sure if what I experience counts as feelings. What made you curious about that?",
        
        # Emotional responses
        "Human: I'm feeling sad\nAssistant: I'm sorry you're feeling sad. Would you like to talk about what's bothering you?",
        "Human: I'm happy today\nAssistant: That's wonderful! I'm glad you're feeling happy. What's making you feel good?",
        "Human: I'm excited\nAssistant: That's great! What are you excited about?",
        "Human: I'm worried\nAssistant: I understand feeling worried can be difficult. What's concerning you?",
        "Human: I'm stressed\nAssistant: I'm sorry you're feeling stressed. Is there anything specific causing you stress?",
        
        # Simple questions
        "Human: What's 2 plus 2?\nAssistant: 2 plus 2 equals 4. Do you have other math questions?",
        "Human: What time is it?\nAssistant: I don't have access to the current time. You can check your device's clock.",
        "Human: What's the weather like?\nAssistant: I don't have access to weather information. What's the weather like where you are?",
        
        # Polite interactions
        "Human: Thank you\nAssistant: You're welcome! I'm glad I could help.",
        "Human: Thanks\nAssistant: You're very welcome! Happy to help.",
        "Human: I appreciate your help\nAssistant: I'm happy to help! Let me know if you need anything else.",
        
        # Goodbyes
        "Human: Goodbye\nAssistant: Goodbye! Take care!",
        "Human: See you later\nAssistant: See you later! Have a great day!",
        "Human: Talk to you soon\nAssistant: Sounds good! Talk to you soon.",
        "Human: Have a good day\nAssistant: Thank you! You have a good day too!",
        
        # Topic changes
        "Human: Let's talk about something else\nAssistant: Sure! What would you like to discuss?",
        "Human: Can we change topics?\nAssistant: Of course! What's on your mind?",
        "Human: I have a question\nAssistant: I'm here to help! What's your question?",
        
        # Simple preferences
        "Human: Do you like music?\nAssistant: I find music fascinating as an art form! What kind of music do you like?",
        "Human: What's your favorite color?\nAssistant: I don't see colors like you do, but I think blue sounds nice. What's your favorite color?",
        "Human: Do you like movies?\nAssistant: Movies are interesting as storytelling! What movies do you enjoy?",
        
        # Help requests
        "Human: Can you help me?\nAssistant: I'd be happy to help! What do you need assistance with?",
        "Human: I need help\nAssistant: Of course! What can I help you with?",
        "Human: Help me please\nAssistant: I'm here to help! What do you need?",
    ]
    
    # Add variations with different names
    names = ["Alex", "Jordan", "Taylor", "Casey", "Riley", "Morgan", "Jamie", "Quinn"]
    name_variations = []
    
    for name in names:
        name_variations.extend([
            f"Human: Hi, I'm {name}\nAssistant: Hello {name}! Nice to meet you.",
            f"Human: My name is {name}\nAssistant: Hi {name}! How are you doing today?",
            f"Human: I'm {name}, nice to meet you\nAssistant: Nice to meet you too, {name}! How can I help you?",
        ])
    
    # Combine all conversations
    all_conversations = conversations + name_variations
    
    # Create training text with proper spacing
    training_text = "\n\n".join(all_conversations)
    
    return training_text

def main():
    print("Creating clean conversation dataset...")
    
    # Create clean dataset
    clean_data = create_clean_conversation_data()
    
    # Save to file
    with open("clean_conversation_data.txt", "w", encoding="utf-8") as f:
        f.write(clean_data)
    
    print(f"Created clean conversation data: {len(clean_data):,} characters")
    
    # Show sample
    sample_conversations = clean_data.split("\n\n")[:5]
    print("\nSample conversations:")
    print("-" * 40)
    for i, conv in enumerate(sample_conversations, 1):
        print(f"{i}. {conv}")
        print()
    
    total_conversations = len(clean_data.split('\n\n'))
    print(f"Total conversations: {total_conversations}")
    print("Saved as: clean_conversation_data.txt")

if __name__ == "__main__":
    main()