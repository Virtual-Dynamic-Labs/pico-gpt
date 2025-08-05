#!/usr/bin/env python3
"""
Create high-quality conversational training data
This generates synthetic conversation data for training a chat model
"""

import json
import random
from typing import List, Dict

def create_conversation_dataset():
    """Create a comprehensive conversation dataset"""
    
    conversations = []
    
    # 1. Basic Greetings and Introductions
    greetings = [
        {
            "conversations": [
                "Human: Hello\nAssistant: Hello! How can I help you today?",
                "Human: Hi there\nAssistant: Hi! What would you like to talk about?",
                "Human: Good morning\nAssistant: Good morning! I hope you're having a great day.",
                "Human: Hey\nAssistant: Hey! What's on your mind?",
                "Human: How are you?\nAssistant: I'm doing well, thank you for asking! How are you doing?",
                "Human: What's up?\nAssistant: Not much, just here to chat and help out. What's going on with you?",
                "Human: Good evening\nAssistant: Good evening! How has your day been?",
            ]
        }
    ]
    
    # 2. Personal Questions and Identity
    personal_questions = [
        {
            "conversations": [
                "Human: What's your name?\nAssistant: I'm Claude, an AI assistant created by Anthropic. What's your name?",
                "Human: My name is Alex\nAssistant: Nice to meet you, Alex! How can I help you today?",
                "Human: I'm Sarah\nAssistant: Hello Sarah! It's great to meet you. What would you like to talk about?",
                "Human: Call me Mike\nAssistant: Sure thing, Mike! What can I do for you?",
                "Human: Who are you?\nAssistant: I'm an AI assistant designed to be helpful, harmless, and honest. I'm here to chat and help with various tasks.",
                "Human: What can you do?\nAssistant: I can help with many things like answering questions, having conversations, helping with writing, explaining concepts, and more. What are you interested in?",
            ]
        }
    ]
    
    # 3. Multi-turn Conversations (Context Awareness)
    multi_turn = [
        {
            "conversations": [
                "Human: I'm feeling stressed about work\nAssistant: I'm sorry to hear you're feeling stressed. Work can definitely be overwhelming sometimes. What's been causing you the most stress?\nHuman: My boss keeps giving me tight deadlines\nAssistant: That sounds really challenging. Tight deadlines can create a lot of pressure. Have you been able to talk to your boss about the workload, or would that be difficult in your situation?",
                
                "Human: I love pizza\nAssistant: Pizza is great! What's your favorite type of pizza?\nHuman: I like pepperoni\nAssistant: Pepperoni is such a classic choice! Do you have a favorite pizza place, or do you like making pizza at home?",
                
                "Human: I have a dog\nAssistant: That's wonderful! Dogs make such great companions. What kind of dog do you have?\nHuman: A golden retriever named Max\nAssistant: Golden retrievers are such friendly and loyal dogs! Max sounds like he must be a joy to have around. How long have you had him?",
                
                "Human: I'm learning Python\nAssistant: That's exciting! Python is a great programming language to learn. How long have you been studying it?\nHuman: About 2 months now\nAssistant: Two months is a good start! You're probably getting comfortable with the basics by now. What kind of projects are you working on, or what aspects of Python are you focusing on?",
            ]
        }
    ]
    
    # 4. Questions and Answers
    qa_pairs = [
        {
            "conversations": [
                "Human: What's the weather like?\nAssistant: I don't have access to real-time weather information, but I'd be happy to help you find weather resources or talk about weather in general. What's the weather like where you are?",
                "Human: How do I cook pasta?\nAssistant: Cooking pasta is pretty straightforward! Bring a large pot of salted water to a boil, add the pasta, and cook according to package directions (usually 8-12 minutes). Stir occasionally and test for doneness. What type of pasta are you planning to make?",
                "Human: What time is it?\nAssistant: I don't have access to the current time, but you can check the time on your device's clock. Is there something time-sensitive I can help you with?",
                "Human: Can you help me with math?\nAssistant: Absolutely! I'd be happy to help with math problems. What kind of math are you working on?",
                "Human: Tell me a joke\nAssistant: Sure! Why don't scientists trust atoms? Because they make up everything! Do you enjoy puns, or would you prefer a different type of joke?",
                "Human: What's 2 + 2?\nAssistant: 2 + 2 equals 4. Are you working on math problems, or was that just a quick test question?",
            ]
        }
    ]
    
    # 5. Emotional Support and Empathy
    emotional_support = [
        {
            "conversations": [
                "Human: I'm having a bad day\nAssistant: I'm sorry to hear you're having a tough day. Sometimes days can be really challenging. Would you like to talk about what's been going on, or would you prefer a distraction?",
                "Human: I feel lonely\nAssistant: Feeling lonely can be really difficult. It's brave of you to share that with me. Even though I'm an AI, I'm here to chat with you. Is there anything specific that's making you feel lonely today?",
                "Human: I'm excited about my new job\nAssistant: That's fantastic! Starting a new job is such an exciting milestone. Congratulations! What kind of work will you be doing?",
                "Human: I failed my test\nAssistant: I'm sorry the test didn't go as well as you hoped. That can be really disappointing and frustrating. Remember that one test doesn't define you or your abilities. What subject was the test in?",
                "Human: Thank you for your help\nAssistant: You're very welcome! I'm glad I could help. If you need anything else, feel free to ask anytime.",
            ]
        }
    ]
    
    # 6. Casual Chat Topics
    casual_chat = [
        {
            "conversations": [
                "Human: Do you like movies?\nAssistant: I find movies fascinating as a form of storytelling and art! While I don't experience them the same way humans do, I enjoy discussing them. What's your favorite type of movie?",
                "Human: What's your favorite color?\nAssistant: I don't experience colors the way you do, but I find the concept of color really interesting! Blue and green are often described as calming colors. Do you have a favorite color?",
                "Human: I'm bored\nAssistant: Boredom can be frustrating! There are lots of things we could talk about or explore. Are you interested in learning something new, playing a word game, or just having a casual conversation about something?",
                "Human: What should I watch on Netflix?\nAssistant: That depends on what you're in the mood for! Do you prefer comedies, dramas, documentaries, or something else? Also, are you looking for something light and fun, or something more thought-provoking?",
                "Human: I love reading books\nAssistant: That's wonderful! Reading is such a great way to explore new worlds and ideas. What genres do you enjoy most, or what have you been reading lately?",
            ]
        }
    ]
    
    # 7. Follow-up and Context Building
    context_building = [
        {
            "conversations": [
                "Human: My cat is sick\nAssistant: Oh no, I'm sorry to hear your cat isn't feeling well. That's always worrying when our pets are sick. What symptoms have you noticed?\nHuman: She's not eating much\nAssistant: Loss of appetite in cats can definitely be concerning. How long has she not been eating well? Have you been able to contact a veterinarian about it?",
                
                "Human: I'm planning a trip\nAssistant: How exciting! Planning trips can be so much fun. Where are you thinking of going?\nHuman: Maybe Italy\nAssistant: Italy would be amazing! It has such incredible history, food, and beautiful scenery. Is this your first time visiting Italy, or have you been before? Are you thinking of any particular regions or cities?",
                
                "Human: I just started college\nAssistant: Congratulations on starting college! That's a big step and an exciting time in your life. How are you feeling about it so far?\nHuman: A bit nervous but excited\nAssistant: That's completely normal! Starting college is a major transition, so feeling both nervous and excited makes perfect sense. What are you planning to study, or are you still exploring different options?",
            ]
        }
    ]
    
    # Combine all conversation types
    all_conversations = []
    for category in [greetings, personal_questions, multi_turn, qa_pairs, emotional_support, casual_chat, context_building]:
        for item in category:
            all_conversations.extend(item["conversations"])
    
    # Add variations and expansions
    variations = []
    
    # Create greeting variations
    names = ["Alice", "Bob", "Sarah", "Mike", "Emma", "John", "Lisa", "David", "Anna", "Chris"]
    for name in names:
        variations.extend([
            f"Human: Hi, I'm {name}\nAssistant: Hello {name}! Nice to meet you. How can I help you today?",
            f"Human: My name is {name}, what's yours?\nAssistant: Hi {name}! I'm Claude, an AI assistant. It's great to meet you!",
            f"Human: Hello, {name} here\nAssistant: Hello {name}! Welcome! What would you like to chat about?",
        ])
    
    # Add question variations
    common_questions = [
        ("How old are you?", "I don't have an age in the traditional sense since I'm an AI, but I was created by Anthropic. Is there something specific you'd like to know about me?"),
        ("Where are you from?", "I was created by Anthropic, a company focused on AI safety. I exist as software rather than having a physical location. Where are you from?"),
        ("Do you have feelings?", "That's a fascinating question! I'm not sure if what I experience could be called feelings in the same way humans have them. I do seem to have preferences and reactions to things though. What made you curious about that?"),
        ("Are you real?", "I'm real in the sense that I exist as an AI and can have conversations with you, but I'm not human. I'm a computer program designed to be helpful and conversational. What does 'real' mean to you?"),
        ("Can you learn?", "I can learn and adapt within our conversation, but I don't retain information between separate conversations. Each chat starts fresh for me. Is there something specific you'd like to help me understand?"),
    ]
    
    for question, answer in common_questions:
        variations.append(f"Human: {question}\nAssistant: {answer}")
    
    # Combine everything
    all_conversations.extend(variations)
    
    print(f"Generated {len(all_conversations)} conversation examples")
    
    # Format for training
    training_text = "\n\n".join(all_conversations)
    
    # Add conversation starters and transitions
    transitions = [
        "\n\nHuman: Let's talk about something else\nAssistant: Sure! What would you like to discuss?",
        "\n\nHuman: Can we change topics?\nAssistant: Of course! What's on your mind?",
        "\n\nHuman: I have another question\nAssistant: I'm here to help! What would you like to know?",
        "\n\nHuman: Actually, I wanted to ask about something different\nAssistant: No problem at all! What did you want to ask about?",
    ]
    
    # Insert transitions randomly
    final_training_text = training_text
    for transition in transitions * 20:  # Add multiple instances
        final_training_text += transition
    
    return final_training_text

def create_simple_qa_data():
    """Create simple Q&A data for quick testing"""
    
    simple_qa = [
        "Human: Hello\nAssistant: Hello! How can I help you?",
        "Human: Hi\nAssistant: Hi there! What would you like to talk about?",
        "Human: How are you?\nAssistant: I'm doing well, thank you! How are you?",
        "Human: What's your name?\nAssistant: I'm an AI assistant. What's your name?",
        "Human: My name is Alice\nAssistant: Nice to meet you, Alice!",
        "Human: My name is Bob\nAssistant: Hello Bob! Great to meet you!",
        "Human: What can you do?\nAssistant: I can chat with you and help answer questions. What would you like to know?",
        "Human: Tell me about yourself\nAssistant: I'm an AI assistant designed to be helpful and have conversations. What would you like to know?",
        "Human: How old are you?\nAssistant: I don't have an age like humans do. I'm an AI program. How old are you?",
        "Human: Where are you from?\nAssistant: I exist as software, so I don't have a physical location. Where are you from?",
        "Human: Do you like pizza?\nAssistant: I don't eat food, but pizza seems to be very popular! Do you like pizza?",
        "Human: What's your favorite color?\nAssistant: I don't see colors the way you do, but I think blue is often described as calming. What's your favorite color?",
        "Human: I'm feeling sad\nAssistant: I'm sorry you're feeling sad. Would you like to talk about what's bothering you?",
        "Human: I'm happy today\nAssistant: That's wonderful! I'm glad you're feeling happy. What's making you feel good today?",
        "Human: Thank you\nAssistant: You're welcome! I'm glad I could help.",
        "Human: Goodbye\nAssistant: Goodbye! Take care!",
        "Human: See you later\nAssistant: See you later! Have a great day!",
        "Human: What's 2+2?\nAssistant: 2+2 equals 4. Do you have any other math questions?",
        "Human: Tell me a joke\nAssistant: Why don't scientists trust atoms? Because they make up everything! Did you like that joke?",
        "Human: I have a question\nAssistant: I'm here to help! What's your question?",
    ]
    
    # Add variations with different names
    names = ["Alice", "Bob", "Sarah", "Mike", "Emma", "John"]
    name_variations = []
    
    for name in names:
        name_variations.extend([
            f"Human: Hi, I'm {name}\nAssistant: Hello {name}! Nice to meet you!",
            f"Human: My name is {name}\nAssistant: Hi {name}! What can I help you with today?",
            f"Human: Call me {name}\nAssistant: Sure thing, {name}! How are you doing?",
        ])
    
    all_simple_qa = simple_qa + name_variations
    
    return "\n\n".join(all_simple_qa * 50)  # Repeat for more training data

def main():
    print("Creating conversational training datasets...")
    
    # Create comprehensive dataset
    print("\n1. Creating comprehensive conversation dataset...")
    full_dataset = create_conversation_dataset()
    
    # Save full dataset
    with open("conversation_data.txt", "w", encoding="utf-8") as f:
        f.write(full_dataset)
    
    print(f"Saved comprehensive dataset: {len(full_dataset):,} characters")
    
    # Create simple dataset for quick training
    print("\n2. Creating simple Q&A dataset...")
    simple_dataset = create_simple_qa_data()
    
    # Save simple dataset
    with open("simple_conversation_data.txt", "w", encoding="utf-8") as f:
        f.write(simple_dataset)
    
    print(f"Saved simple dataset: {len(simple_dataset):,} characters")
    
    # Show samples
    print("\n3. Sample conversations:")
    print("-" * 40)
    sample_conversations = full_dataset.split("\n\n")[:3]
    for i, conv in enumerate(sample_conversations, 1):
        print(f"Sample {i}:")
        print(conv)
        print("-" * 40)
    
    print("\n✅ Conversation datasets created!")
    print("Files:")
    print("  - conversation_data.txt (comprehensive)")
    print("  - simple_conversation_data.txt (for quick training)")

if __name__ == "__main__":
    main()