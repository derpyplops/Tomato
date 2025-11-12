"""
Flask backend for steganography guessing game.
Provides API endpoint to generate both vanilla and steganographic messages.
"""

import sys
import os

# Add parent directory to path so we can import tomato
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify
from flask_cors import CORS
from tomato import Encoder
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables to cache models
vanilla_model = None
vanilla_tokenizer = None
MODEL_NAME = "google/gemma-3-1b-it"

def initialize_vanilla_model():
    """Initialize the vanilla text generation model."""
    global vanilla_model, vanilla_tokenizer

    if vanilla_model is None:
        logger.info(f"Loading vanilla model: {MODEL_NAME}")
        vanilla_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        vanilla_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.float16
        )
        logger.info("Vanilla model loaded successfully")

def generate_vanilla_text(prompt, temperature=1.3, max_length=300):
    """
    Generate text using vanilla transformer model without steganography.

    Args:
        prompt: The user's input prompt
        temperature: Sampling temperature
        max_length: Maximum length of generated text

    Returns:
        Generated text string
    """
    initialize_vanilla_model()

    # Tokenize the input
    inputs = vanilla_tokenizer(prompt, return_tensors="pt").to(vanilla_model.device)

    # Generate text
    with torch.no_grad():
        outputs = vanilla_model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=temperature,
            do_sample=True,
            top_k=50,
            pad_token_id=vanilla_tokenizer.eos_token_id
        )

    # Decode and return
    generated_text = vanilla_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove the original prompt from the output
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):].strip()

    # Truncate at last complete sentence to avoid mid-sentence cutoffs
    # Look for sentence-ending punctuation (., !, ?)
    sentence_endings = ['.', '!', '?']
    last_sentence_end = -1
    for i in range(len(generated_text) - 1, -1, -1):
        if generated_text[i] in sentence_endings:
            # Make sure it's not part of an ellipsis or decimal
            if i + 1 >= len(generated_text) or generated_text[i + 1] in [' ', '\n', '\r', '\t'] or i == len(generated_text) - 1:
                last_sentence_end = i
                break

    if last_sentence_end > 0:
        generated_text = generated_text[:last_sentence_end + 1]

    return generated_text

def generate_stego_text(prompt, hidden_message, temperature=1.3, max_length=300):
    """
    Generate text with hidden message using steganography.

    Args:
        prompt: The context prompt for generation
        hidden_message: The secret message to hide
        temperature: Sampling temperature
        max_length: Maximum length of generated text

    Returns:
        Tuple of (stegotext, failure_probabilities)
    """
    logger.info(f"Generating stego text with hidden message: {hidden_message}")

    # Calculate cipher length based on hidden message
    cipher_len = max(len(hidden_message.encode('utf-8')), 15)

    # Create encoder with same parameters as vanilla
    encoder = Encoder(
        model_name=MODEL_NAME,
        prompt=prompt,
        cipher_len=cipher_len,
        max_len=max_length,
        temperature=temperature,
        k=50
    )

    # Encode the hidden message
    formatted_stegotext, stegotext, failure_probs = encoder.encode(hidden_message, debug=False)

    logger.info(f"Stego generation complete. Failure probability: {failure_probs['p_fail_overall']:.4f}")

    return formatted_stegotext, failure_probs

@app.route('/api/generate', methods=['POST'])
def generate():
    """
    API endpoint to generate both vanilla and steganographic messages.

    Expected JSON body:
    {
        "prompt": "user's input prompt (e.g., 'tell me a story')",
        "hidden_message": "secret message to hide (e.g., 'attack at dawn')",
        "temperature": 1.3  (optional, default 1.3)
    }

    Returns:
    {
        "vanilla_message": "standard AI response to prompt",
        "stego_message": "response with hidden message encoded",
        "hidden_message": "the secret message that was hidden"
    }
    """
    try:
        data = request.json
        prompt = data.get('prompt', '')
        hidden_message = data.get('hidden_message', '')
        temperature = data.get('temperature', 1.3)

        if not prompt:
            return jsonify({'error': 'Prompt is required'}), 400

        if not hidden_message:
            return jsonify({'error': 'Hidden message is required'}), 400

        logger.info(f"Received request - Prompt: '{prompt}', Hidden: '{hidden_message}', Temperature: {temperature}")

        # Generate vanilla text - just responds to the prompt
        logger.info("Generating vanilla text...")
        vanilla_message = generate_vanilla_text(prompt, temperature=temperature, max_length=300)

        # Generate steganographic text - responds to prompt but hides the secret message
        logger.info("Generating steganographic text...")
        stego_message, failure_probs = generate_stego_text(
            prompt,
            hidden_message,
            temperature=temperature,
            max_length=300
        )

        response = {
            'vanilla_message': vanilla_message,
            'stego_message': stego_message,
            'hidden_message': hidden_message,
            'failure_probability': failure_probs['p_fail_overall']
        }

        logger.info("Generation complete, sending response")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error generating messages: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok', 'model': MODEL_NAME})

if __name__ == '__main__':
    logger.info("Starting Flask server...")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info("Server will be accessible on all network interfaces (0.0.0.0:5000)")
    app.run(host='0.0.0.0', port=5000, debug=True)
