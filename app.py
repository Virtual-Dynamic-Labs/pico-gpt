#!/usr/bin/env python3
"""
Gradio app for Pico GPT Conversational Model
Deployable on Hugging Face Spaces for free
"""

import gradio as gr
import torch
import os
import sys

# Add src to path
sys.path.append('./src')
from pico_gpt import GPT, GPTConfig
from tokenizer import SimpleTokenizer

class PicoGPTChat:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.config = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.load_model()
    
    def load_model(self):
        """Load the model and tokenizer"""
        try:
            # Load from Hugging Face hub or local file
            model_path = 'pico_gpt_conversation.pt'
            if not os.path.exists(model_path):
                # Try to download from HF hub
                from huggingface_hub import hf_hub_download
                model_path = hf_hub_download(
                    repo_id="jemico/pico-gpt-conversational", 
                    filename="pico_gpt_conversation.pt"
                )
            
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            self.config = checkpoint['config']
            self.tokenizer = checkpoint['tokenizer']
            
            self.model = GPT(self.config)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ Model loaded on {self.device}")
            print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            print(f"   Validation loss: {checkpoint['best_val_loss']:.4f}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise e
    
    def generate_response(self, user_input, temperature=0.7, max_tokens=50):
        """Generate a response to user input"""
        if not self.model or not user_input.strip():
            return "Please enter a message!"
        
        try:
            # Format input
            prompt = f"Human: {user_input.strip()}\nAssistant:"
            
            # Tokenize
            tokens = self.tokenizer.encode(prompt)
            context = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)
            
            # Generate
            with torch.no_grad():
                generated = self.model.generate(
                    context,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_k=15
                )
            
            # Decode and extract response
            full_text = self.tokenizer.decode(generated[0].tolist())
            
            if "Assistant" in full_text:
                parts = full_text.split("Assistant")
                if len(parts) > 1:
                    response = parts[1].strip()
                    
                    # Clean up response
                    for stop in ["Human", "---", "\n\n---", "\n---"]:
                        if stop in response:
                            response = response.split(stop)[0].strip()
                            break
                    
                    response = response.rstrip("'\"")
                    return response if response else "I'm not sure how to respond to that."
            
            return "I'm having trouble generating a response. Please try rephrasing."
            
        except Exception as e:
            return f"Error: {str(e)}"

# Initialize the model
chat_bot = PicoGPTChat()

def chat_interface(message, history, temperature, max_tokens):
    """Gradio chat interface"""
    if not message.strip():
        return history, history, ""
    
    # Generate response
    response = chat_bot.generate_response(message, temperature, max_tokens)
    
    # Update history
    history.append([message, response])
    
    return history, history, ""

# Create Gradio interface
with gr.Blocks(title="🤖 Pico GPT Conversational AI") as demo:
    gr.Markdown("""
    # 🤖 Pico GPT Conversational AI
    
    A lightweight conversational AI model trained for natural dialogue interactions.
    
    **Model Stats:**
    - 26M parameters
    - 256 token context
    - Trained on curated conversation data
    - Optimized for single-turn responses
    """)
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                value=[],
                label="💬 Conversation",
                height=400,
                show_label=True
            )
            
            msg = gr.Textbox(
                label="Your message",
                placeholder="Type your message here...",
                lines=2,
                max_lines=5
            )
            
            with gr.Row():
                send_btn = gr.Button("Send 🚀", variant="primary")
                clear_btn = gr.Button("Clear 🗑️")
        
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Settings")
            
            temperature = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.7,
                step=0.1,
                label="🌡️ Temperature",
                info="Higher = more creative"
            )
            
            max_tokens = gr.Slider(
                minimum=20,
                maximum=100,
                value=50,
                step=10,
                label="📏 Max Tokens",
                info="Response length"
            )
            
            gr.Markdown("""
            ### 💡 Tips
            - Keep messages conversational
            - Try greetings, questions, requests
            - Each response is independent
            - Model works best with simple queries
            """)
    
    # Event handlers
    send_btn.click(
        chat_interface,
        inputs=[msg, chatbot, temperature, max_tokens],
        outputs=[chatbot, chatbot, msg]
    )
    
    msg.submit(
        chat_interface,
        inputs=[msg, chatbot, temperature, max_tokens],
        outputs=[chatbot, chatbot, msg]
    )
    
    clear_btn.click(
        lambda: ([], ""),
        outputs=[chatbot, msg]
    )
    
    gr.Markdown("""
    ---
    **Model:** [jemico/pico-gpt-conversational](https://huggingface.co/jemico/pico-gpt-conversational) | 
    **License:** MIT | 
    **Framework:** PyTorch + Gradio
    """)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )