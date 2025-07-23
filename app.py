import os
import torch
import random
import logging
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configure basic logging for visibility during startup and operations
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Global variables for models and tokenizers
finetuned_model = None
finetuned_tokenizer = None
original_model = None # For before/after comparison (will be None in this version)
original_tokenizer = None # For before/after comparison (will be None in this version)

# Use absolute path for robustness, assuming finetuned_model is in the same directory as app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
finetuned_model_path = os.path.join(BASE_DIR, "finetuned_model")
original_model_name = "distilgpt2"
status_message = "Models not loaded yet."

@app.before_request
def load_models_on_startup():
    """
    Loads models and tokenizers when the Flask app starts.
    This version prioritizes loading only the fine-tuned model for memory efficiency.
    """
    global finetuned_model, finetuned_tokenizer, original_model, original_tokenizer, status_message

    if finetuned_model is None: # Only load if not already loaded
        try:
            # --- Device Selection (FORCED TO CPU for stability on Mac) ---
            # Using "cpu" explicitly to avoid potential MPS issues during model loading/inference.
            # This is crucial for systems with limited RAM or MPS compatibility issues.
            device = "cpu"
            logging.info(f"Attempting to load models to device: {device} for stability.")

            # --- Load Fine-tuned Model ---
            # Explicitly check for pytorch_model.bin to confirm finetuning completion
            if not os.path.exists(finetuned_model_path) or not os.path.exists(os.path.join(finetuned_model_path, 'pytorch_model.bin')):
                status_message = f"Fine-tuned model files not found at {finetuned_model_path}. Please run 'python3 finetune_model.py' first."
                logging.error(status_message) # Using logging for better visibility
                return

            logging.info(f"Loading fine-tuned model from {finetuned_model_path}...")
            finetuned_tokenizer = AutoTokenizer.from_pretrained(finetuned_model_path)
            # Move model to selected device and load in half-precision (float16) for memory efficiency
            finetuned_model = AutoModelForCausalLM.from_pretrained(finetuned_model_path, torch_dtype=torch.float16).to(device)
            finetuned_model.eval() # Ensure model is in evaluation mode
            logging.info("Fine-tuned model loaded to device.")

            # --- Load Original Model (COMMENTED OUT FOR MEMORY EFFICIENCY) ---
            # If you want to re-enable this, uncomment the block below.
            # Remember to also adjust the 'highlight_difference' route if you do.
            # logging.info(f"Loading original model '{original_model_name}' (for comparison)...")
            # original_tokenizer = AutoTokenizer.from_pretrained(original_model_name)
            # if "<|newline|>" not in original_tokenizer.vocab:
            #     original_tokenizer.add_special_tokens({'additional_special_tokens': ['<|newline|>']})
            # original_tokenizer.pad_token = original_tokenizer.eos_token
            # original_model = AutoModelForCausalLM.from_pretrained(original_model_name, torch_dtype=torch.float16).to(device)
            # original_model.resize_token_embeddings(len(original_tokenizer))
            # original_model.eval()
            # logging.info("Original model loaded to device.")

            status_message = "Fine-tuned model loaded successfully." # Adjusted status message
            logging.info(status_message)

        except Exception as e:
            status_message = f"CRITICAL ERROR loading models: {str(e)}"
            logging.exception(status_message) # Logs the full traceback for detailed error info
            if "out of memory" in str(e).lower():
                status_message += ". This likely indicates a memory (RAM) issue during model loading. Try freeing up RAM or reducing batch sizes in training."
            elif "mps" in str(e).lower() and "fail" in str(e).lower():
                status_message += ". This suggests an issue with Apple Silicon (MPS) acceleration. Forcing CPU might help (which this code already does)."
            logging.error(status_message)


def generate_text(model, tokenizer, start_string, num_generate=50, temperature=0.7):
    """
    Generates text using the given model and tokenizer.
    Returns a single generated string, stopping at first newline or max length.
    """
    # Ensure start_string is a string
    if not isinstance(start_string, str):
        start_string = str(start_string)

    # Force device to CPU for input_ids as well, matching model device
    device = "cpu"

    if start_string:
        input_ids = tokenizer.encode(start_string, return_tensors='pt').to(device)
    else:
        if not tokenizer.vocab:
            return "" # Cannot generate without a vocabulary
        
        possible_starts = [t for t in tokenizer.vocab.keys() if len(t) == 1 and t.isalpha()]
        if not possible_starts:
            possible_starts = list(tokenizer.vocab.keys())
        
        random_start_char = random.choice(possible_starts)
        input_ids = tokenizer.encode(random_start_char, return_tensors='pt').to(device)


    # Generate text
    output = model.generate(
        input_ids,
        max_length=num_generate + len(input_ids[0]), # Total max length including prompt
        num_return_sequences=1,
        do_sample=True,
        temperature=temperature,
        top_k=50,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    generated_sequence = tokenizer.decode(output[0], skip_special_tokens=False)

    # --- START OF IMPROVED POST-PROCESSING ---
    # Post-process to extract a single coherent name
    name = generated_sequence.strip()

    # Find delimiters in order of preference
    delimiters = ["<|newline|>", "\n", ":", ",", ".", " "] # Added ':' and ',' as strong delimiters

    for delim in delimiters:
        idx = name.find(delim)
        if idx != -1:
            name = name[:idx].strip() # Cut off at the delimiter
            break # Stop after finding the first one

    # Remove any lingering special newline token if it wasn't the first delimiter
    name = name.replace("<|newline|>", "").strip()

    # Ensure the name does not contain the colon or comma
    name = name.split(':')[0].split(',')[0].strip()

    # Re-add prompt if it was removed or not correctly prefixed
    if start_string and not name.lower().startswith(start_string.lower()):
        name = start_string + name

    # Basic cleaning: remove extra spaces and limit to reasonable length if still too long
    name = ' '.join(name.split()) # Normalize spaces
    name = name[:num_generate] # Hard truncate to max requested length if it's still too long

    # Final check: if it's still very long (and a prompt was given),
    # try to find the first few words to ensure it's name-like.
    if len(name) > (len(start_string) + 20) and ' ' in name: 
        name_parts = name.split(' ')
        if start_string and len(start_string.split()) > 0:
            prompt_word_count = len(start_string.split())
            name = ' '.join(name_parts[:prompt_word_count + 2])
        else:
            name = ' '.join(name_parts[:3])
        name = name.strip()

    # Ensure name is not empty
    if not name:
        name = start_string if start_string else "GeneratedName" # Fallback if post-processing made it empty
        
    return name
    # --- END OF IMPROVED POST-PROCESSING ---

@app.route('/')
def home():
    return "Welcome to the Fantasy Name Generator API! Use /generate or /status."

@app.route('/generate', methods=['GET'])
def generate_name():
    """
    Generates fantasy names using the fine-tuned model.
    Query parameters:
        prompt (str, optional): The starting string for name generation.
        num_names (int, optional): Number of names to generate (default: 1, max: 5).
        length (int, optional): Max length of generated name (default: 30, max: 60).
        temperature (float, optional): Controls randomness (default: 0.8, range: 0.1 to 1.5).
    """
    if finetuned_model is None:
        return jsonify({"error": status_message, "hint": "Run 'python3 finetune_model.py' first."}), 500

    prompt = request.args.get('prompt', '').strip()
    
    try:
        num_names = int(request.args.get('num_names', 1))
        length = int(request.args.get('length', 30))
        temperature = float(request.args.get('temperature', 0.8))
    except ValueError:
        return jsonify({"error": "num_names, length, and temperature must be valid numbers."}), 400

    # Input validation
    if not (0.1 <= temperature <= 1.5): # GPT-like models can get chaotic with high temp
        return jsonify({"error": "Temperature must be between 0.1 and 1.5."}), 400
    if not (1 <= num_names <= 5): # Limit for practical reasons for API calls
        return jsonify({"error": "num_names must be between 1 and 5."}), 400
    if not (10 <= length <= 60): # Max name length
        return jsonify({"error": "length must be between 10 and 60."}), 400

    generated_names = []
    for _ in range(num_names):
        name = generate_text(finetuned_model, finetuned_tokenizer, prompt, num_generate=length, temperature=temperature)
        generated_names.append(name)

    return jsonify({"names": generated_names})

@app.route('/highlight_difference', methods=['GET'])
def highlight_difference():
    """
    Generates names using both the original and fine-tuned models for comparison.
    Query parameters:
        prompt (str, optional): The starting string for name generation.
        num_names (int, optional): Number of names to generate (default: 1, max: 3).
        length (int, optional): Max length of generated name (default: 30).
        temperature (float, optional): Controls randomness (default: 0.8).
    """
    # Adjust this check: only require finetuned_model to be loaded
    if finetuned_model is None:
        return jsonify({"error": status_message, "hint": "Fine-tuned model not loaded. Original model comparison disabled for memory optimization."}), 500

    prompt = request.args.get('prompt', 'The').strip() # Default prompt for comparison
    
    try:
        num_names = int(request.args.get('num_names', 1))
        length = int(request.args.get('length', 30))
        temperature = float(request.args.get('temperature', 0.8))
    except ValueError:
        return jsonify({"error": "num_names, length, and temperature must be valid numbers."}), 400

    if not (1 <= num_names <= 3):
        return jsonify({"error": "num_names for comparison must be between 1 and 3."}), 400

    results = {
        "finetuned_model_names": [] # Only fine-tuned names will be generated
    }

    # Only generate original names if original_model was actually loaded (which it won't be in this version)
    if original_model is not None:
        results["original_model_names"] = []
        for i in range(num_names):
            original_name = generate_text(original_model, original_tokenizer, prompt, num_generate=length, temperature=temperature)
            results["original_model_names"].append(original_name)
    else:
        logging.warning("Original model not loaded. Comparison will only show fine-tuned results.")
        results["original_model_names"] = ["Original model not loaded (memory optimization)."] * num_names


    for i in range(num_names):
        finetuned_name = generate_text(finetuned_model, finetuned_tokenizer, prompt, num_generate=length, temperature=temperature)
        results["finetuned_model_names"].append(finetuned_name)

    return jsonify(results)

@app.route('/status', methods=['GET'])
def get_status():
    """
    Provides information about the server and loaded models.
    """
    return jsonify({
        "server_status": "running",
        "model_load_status": status_message,
        "finetuned_model_loaded": finetuned_model is not None,
        "original_model_loaded": original_model is not None, # This will likely be False in this version
        "model_name_fine_tuned": finetuned_model_path,
        "model_name_original": original_model_name
    })

if __name__ == '__main__':
    # Set use_reloader=False if you have @app.before_request to avoid double-loading models
    # during Flask's debug auto-reloading feature, which can be memory intensive.
    print("Starting Flask API...")
    app.run(debug=True, host='0.0.0.0', port=5001)