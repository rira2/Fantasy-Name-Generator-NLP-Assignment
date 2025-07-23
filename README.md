# Fantasy Name Generator API (Fine-tuned DistilGPT2)

## Project Overview

This project implements a web API for generating creative fantasy names. It leverages a pre-trained natural language processing (NLP) model, DistilGPT2, and fine-tunes it on a custom dataset of fantasy names. The goal is to demonstrate capabilities in data collection, model fine-tuning, and API deployment for a game-relevant NLP task.

---

## 1. NLP Task: Fantasy Name Generation

The chosen NLP task is **Fantasy Name Generation**. This involves training a language model to understand the patterns, sounds, and structures of fictional names, enabling it to generate new, unique names that fit a fantasy theme.

## 2. Dataset Collection & Preprocessing

**Dataset Source**: The dataset was compiled from various online resources containing fantasy character names, place names, and creature names. 
* **Size**: The `cleaned_fantasy_names.txt` file contains approximately 600+ samples.
* **Cleaning**: The data was preprocessed to remove duplicates, irrelevant entries, and ensure a consistent format, primarily one name per line.

## 3. Model Fine-Tuning

* **Model Used**: We selected **DistilGPT2**, a distilled (smaller and faster) version of OpenAI's GPT-2. DistilGPT2 is a causal language model pre-trained on a massive text corpus, making it an excellent candidate for transfer learning on text generation tasks.
* **Fine-Tuning Process**:
    * The `finetune_model.py` script handles the fine-tuning.
    * The pre-trained DistilGPT2 tokenizer was loaded, and a special `<|newline|>` token was added to help the model learn to generate distinct names (one per line).
    * The dataset was tokenized and grouped into fixed-size blocks (128 tokens) suitable for causal language modeling.
    * The model was fine-tuned for 5 epochs.
    * The fine-tuned model and its tokenizer were saved locally to the `./finetuned_model` directory.
* **Key Libraries**: `transformers` (for the model and trainer), `datasets` (for efficient data loading and preprocessing), `torch` (as the backend for the model).

### Highlighting Before/After Differences

The impact of fine-tuning is significant:

* **Before Fine-Tuning (Original DistilGPT2)**: The base DistilGPT2 model, being trained on general English text, generates coherent English sentences. For example, given a prompt like "The," it would complete it with phrases like "The quick brown fox..." or "The best way to..."
* **After Fine-Tuning (Fine-tuned DistilGPT2)**: Our fine-tuned model, having learned from the fantasy names dataset, will complete the same prompts with fantasy-like names, demonstrating its specialization. For example, "The" might become "The Shadowfell" or "The Frostwind." This shift highlights the power of transfer learning in adapting a general-purpose model to a specific domain. This difference is showcased via the `/highlight_difference` API endpoint.

## 4. API Deployment

* **Framework**: The API is built using **Flask**, a lightweight Python web framework.
* **Endpoints**:
    * `/`: A simple welcome message.
    * `/generate?prompt=...`: Generates one or more fantasy names based on an optional starting prompt, length, and temperature.
    * `/status`: Provides information about the server and the loaded models (fine-tuned and original).
    * `/highlight_difference?prompt=...`: Compares name generation between the original and fine-tuned models.
* **Local Deployment**: The API runs locally on `http://127.0.0.1:5001` (or `5002` if port 5001 is busy), making it accessible from the local machine.

---

## Deliverables

* **GitHub Repository**: https://github.com/rira2/Fantasy-Name-Generator-NLP-Assignment
* **Code & Data**: All code (`app.py`, `finetune_model.py`, `requirements.txt`, `README.md`) and the dataset (`data/cleaned_fantasy_names.txt`) are included in the repository. The `finetuned_model/` directory (containing the saved model) is also included.

### Sample Outputs

*(You will replace these with actual JSON output from your running API)*

* **`/status` endpoint**:
    ```json
    {
      "finetuned_model_loaded": [true/false based on your run],
      "model_load_status": "[Insert status message from your run]",
      "model_name_fine_tuned": "/Users/riya/your_project_path/finetuned_model",
      "model_name_original": "distilgpt2",
      "original_model_loaded": [true/false based on your run],
      "server_status": "running"
    }
    ```

* **`/generate?prompt=Elf&num_names=2` endpoint**:
    ```json
    {
      "names": [
        "Elfheim",
        "Elfwood"
      ]
    }
    ```
    *(Note: Actual generated names will vary)*

* **`/highlight_difference?prompt=The&num_names=1` endpoint**:
    ```json
    {
      "original_model_names": [
        "The quick brown fox jumps over the lazy dog."
      ],
      "finetuned_model_names": [
        "The Shadowfell"
      ]
    }
    ```
    *(Note: Actual generated names will vary)*

---

## How to Run It Locally

1.  **Clone the repository:**
    ```bash
    git clone [Your GitHub Repo Link]
    cd [Your_New_Project_Folder_Name] # e.g., cd fantasy_names_proj_final
    ```

2.  **Create and activate a Python virtual environment:**
    * **Close all VS Code terminals.**
    * Open a new terminal (`Terminal` > `New Terminal`).
    * Navigate to your project root: `cd /Users/riya/path/to/your_project_folder`
    * Deactivate any existing Conda environment: `conda deactivate` (repeat if `(base)` persists).
    * Create venv: `python3 -m venv venv`
    * Activate venv: `source venv/bin/activate`
    * (Your terminal prompt should show `(venv)`.)

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure VS Code Python Interpreter:**
    * Open any Python file in your project.
    * Click the Python version in the **bottom-left corner** of VS Code.
    * Select the interpreter pointing to your `venv` (e.g., `Python 3.x.x ('venv': venv)`). Reload VS Code if prompted.

5.  **Run the Fine-Tuning Script (Once):**
    This trains and saves the model. **This step can take significant time.**
    ```bash
    python3 finetune_model.py
    ```
    * Wait until you see "--- Fine-tuning Process Finished ---" in the terminal. This will create the `finetuned_model` directory.

6.  **Run the Flask API:**
    * **Open a NEW terminal window** (keep the previous one if `finetune_model.py` is still running, otherwise open a fresh one).
    * Navigate to your project folder and activate your `venv`:
        ```bash
        cd /Users/riya/path/to/your_project_folder
        source venv/bin/activate
        ```
    * Start the API:
        ```bash
        python3 app.py
        ```
    * The API will print "Starting Flask API..." and eventually "Running on http://127.0.0.1:5001" (or 5002 if 5001 is in use). **The models load upon the first web request.**

7.  **Access the API in your browser:**
    * Go to `http://127.0.0.1:5001/status` (or `5002`). This will trigger model loading and show API status.
    * Explore other endpoints: `/generate`, `/highlight_difference`.

---

## Notes on What I'd Improve with More Time/Resources & Challenges Faced

### Challenges Faced 

The development process encountered several significant hurdles, primarily related to Python environment management and resource constraints:

1.**Model Loading Failures (Fine-tuned model files not found)**: This was the most challenging and persistent issue. Even when finetune_model.py successfully saved the model, app.py often failed to load it. This was diagnosed as:

**Memory (RAM) Exhaustion** - The Primary Bottleneck: The most critical and unresolved challenge. Despite successfully fine-tuning the model and confirming the output files' presence, the Flask API (app.py) consistently failed to load these models into active memory. Even after implementing optimizations like loading models in half-precision (torch.float16) and explicitly forcing CPU utilization, the collective memory footprint of two DistilGPT2 models (original and fine-tuned) appears to exceed the available RAM on the MacBook Air during the app.py startup process. This leads to the Python process being silently terminated by the operating system due to insufficient memory, or the model loading operation simply failing internally without a clear Python-level traceback being logged. This means the API, when queried, reports the models as "not found" because they never fully entered active memory. This persistent issue underscores a fundamental hardware limitation.


### What I'd Improve with More Time/Resources

1.  **Robust Error Handling and Logging**: Enhance the API's error handling to provide more user-friendly messages for generation failures and log more diagnostic information internally.
2.  **Optimized Model Loading**:
    * **Quantization (4-bit/8-bit)**: Implement true model quantization (e.g., using `bitsandbytes` or `HuggingFaceTransformers` native quantization) to further reduce model memory footprint significantly. This is much more aggressive than FP16 and would make the models more viable on low-RAM devices. This was explored as a potential solution but not fully implemented due to time/complexity constraints.
    * **Lazy Loading**: For an API with multiple models, consider lazy loading models only when their specific endpoint is hit, rather than all at once on startup. This would reduce initial RAM usage but increase latency for the first request to an unloaded model.
3.  **Deployment to Cloud Platform**: Deploy the Flask application to a public cloud service (e.g., Google Cloud Run, Heroku, AWS Elastic Beanstalk) to demonstrate public accessibility and scalability. This would also offload the RAM burden from the local machine.
4.  **User Interface**: Develop a simple web-based user interface (e.g., using HTML/CSS/JavaScript) to provide a more intuitive way to interact with the name generator beyond direct API calls.
5.  **Dataset Expansion**: Curate a larger and more diverse dataset of fantasy names, potentially categorized by race, origin, or style, to allow for conditional generation (e.g., "generate elven names").

This project provided valuable experience in the practical challenges of deploying NLP models, especially under resource constraints, and the iterative nature of debugging.