import os
import sys
import numpy as np
import time
from typing import List, Dict, Any, Tuple
# --- SPACY (ADVANCED NER) IMPORTS ---

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
try:
    import spacy
    from spacy import displacy
    SPACY_MODEL_NAME = "en_core_web_sm"
    SPACY_SUPPORT = True
except ImportError:
    # This warning helps the user install spaCy
    print("Warning: spaCy library is missing. Install with 'pip install spacy'.")
    SPACY_SUPPORT = False

# --- TEXT-TO-SPEECH (TTS) IMPORTS ---
try:
    from gtts import gTTS
    import playsound
    TTS_SUPPORT = True
except ImportError:
    print("Warning: gTTS or playsound libraries are missing. Install with 'pip install gTTS playsound'. TTS functionality disabled.")
    TTS_SUPPORT = False

# --- CRITICAL LIBRARY IMPORTS ---
try:
    # Use google.genai and create an alias 'genai' to prevent conflicts
    import google.generativeai as genai_module
    from google.generativeai import types
    genai = genai_module
except ImportError:
    print("FATAL ERROR: The 'google-genai' library is missing. Install it using the command above.")
    genai = None

# --- Multilingual NLP Library Imports (Optional but recommended) ---
try:
    from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
    from sklearn.cluster import KMeans
    from scipy.spatial.distance import cosine
    # We still import these to define the functions, but model loading is managed below
    NER_SUPPORT = True
    CLUSTERING_SUPPORT = True
except ImportError:
    print("Warning: Optional NLP/clustering libraries (transformers, scikit-learn, scipy) are missing.")
    NER_SUPPORT = False
    CLUSTERING_SUPPORT = False
    # Define placeholder objects/functions for fallback
    NER_PIPELINE = None
    KMeans = None
    cosine = None
    def dummy_func_ner(*args, **kwargs):
        return [{'entity': 'Disabled', 'text': 'Missing optional libraries for NER.'}]
    def dummy_func_cat(*args, **kwargs):
        return {"Error": ["Clustering disabled: Missing scikit-learn or other libraries."]}
    def dummy_func_search(*args, **kwargs):
        return [("Semantic Search Disabled", 0.0)]

# Redefine functions to point to dummies if libraries are missing
if not NER_SUPPORT:
    analyze_entities = dummy_func_ner
if not CLUSTERING_SUPPORT:
    categorize_prompts = dummy_func_cat
    semantic_search = dummy_func_search


# -----------------------------------------------------------------------------
# 1. CONFIGURATION & DIRECT MODEL INITIALIZATION
# -----------------------------------------------------------------------------

# Using 'gemini-pro' for broad compatibility to avoid potential 'not found' errors.
MODEL_NAME = 'gemini-2.5-flash'
EMBEDDING_MODEL = 'models/text-embedding-004'
DEFAULT_TEMPERATURE = 0.7

# --- Multilingual Mapping ---
LANGUAGES_MAP = {
    'en': 'English',
    'hi': 'Hindi',
    'te': 'Telugu',
    'ta': 'Tamil',
    'ml': 'Malayalam',
    'kn': 'Kannada'
}

# 🔑 YOUR API KEY: Directly inserted for reliability
API_KEY_DIRECT = ""#put ur API key here" ""

def initialize_gemini_model() -> genai.GenerativeModel | None:
    """Initializes the Gemini GenerativeModel using the direct API key string."""
    # Set the OpenMP environment variable to avoid conflicts (Fix for the KMP_DUPLICATE_LIB_OK warning)
    if sys.platform.startswith('win'):
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    if genai is None:
        print("Cannot initialize Gemini model: google-genai library not imported.")
        return None

    if not API_KEY_DIRECT or API_KEY_DIRECT == "YOUR_API_KEY_HERE":
        print("FATAL ERROR: API Key is missing or default. Please provide a valid key.")
        return None

    try:
        genai.configure(api_key=API_KEY_DIRECT)
        model = genai.GenerativeModel(model_name=MODEL_NAME)
        try:
            model.count_tokens("test")
            print("✅ Gemini Model initialized and key validated successfully.")
        except Exception as e:
            print(f"❌ Warning: Basic model validation failed: {e}")
        return model

    except Exception as e:
        print(f"❌ Error initializing/validating model: {e}")
        return None

# Global model object (initialized once)
gemini_model = initialize_gemini_model()


# --- Spacy Model Loading (for Choice 9) ---
SPACY_NLP = None
if SPACY_SUPPORT:
    try:
        print(f"Loading spaCy model: {SPACY_MODEL_NAME}...")
        SPACY_NLP = spacy.load(SPACY_MODEL_NAME)
        print("✅ spaCy model loaded.")
    except OSError:
        # We handle the model file not being downloaded separately from the library being absent
        print(f"\n❌ Error: spaCy model '{SPACY_MODEL_NAME}' not found.")
        print(f"Please run this command in your terminal: 'python -m spacy download {SPACY_MODEL_NAME}'")
        SPACY_SUPPORT = False


# --- IndicNER Model Loading (Non-LLM component) ---
NER_PIPELINE = None
if NER_SUPPORT:
    try:
        NER_MODEL_NAME = 'ai4bharat/IndicNER'
        # =========================================================================
        # TEMPORARILY COMMENT OUT MODEL LOADING DUE TO PERSISTENT DEPENDENCY CRASHES
        # =========================================================================
        # print("Loading IndicNER model for multilingual NER...")
        # NER_PIPELINE = pipeline(
        #     "ner",
        #     model=AutoModelForTokenClassification.from_pretrained(NER_MODEL_NAME),
        #     tokenizer=AutoTokenizer.from_pretrained(NER_MODEL_NAME),
        #     aggregation_strategy="simple"
        # )
        # print("✅ IndicNER loaded.")

        # Add a placeholder print to confirm we skipped the potentially crashing code
        print(f"⏩ Skipped loading IndicNER model due to previous stability issues.")

    except Exception as e:
        print(f"❌ Error loading IndicNER: {e}")
        NER_PIPELINE = None


# -----------------------------------------------------------------------------
# 2. CORE UTILITY FUNCTIONS (LLM, NLP, and TTS)
# -----------------------------------------------------------------------------

def get_sentence_embeddings(texts: List[str]) -> List[List[float]]:
    """Generates sentence embeddings using the Gemini Embedding model."""
    if genai is None:
        print("Embedding failed: google-genai library not imported.")
        return []
    if not texts: return []
    try:
        response = genai.embed_content(model=EMBEDDING_MODEL, content=texts)
        return response['embedding']
    except Exception as e:
        print(f"Embedding failed: {e}")
        return []

def run_gemini_task(
    system_instruction: str,
    user_prompt: str,
    target_lang_code: str,
    temperature: float = DEFAULT_TEMPERATURE
) -> str:
    """Runs a task, enforcing the output language via system instruction."""
    if gemini_model is None:
        return "LLM Generation Error: Gemini model not initialized."

    target_lang = LANGUAGES_MAP.get(target_lang_code, 'English')

    language_constraint = (
        f"You MUST generate the entire output in {target_lang}. "
        f"Do not include any English or other language text."
    )

    full_prompt = f"{system_instruction}\n\n{language_constraint}\n\n{user_prompt}"

    try:
        generation_config = {
            "temperature": temperature,
        }

        response = gemini_model.generate_content(
            contents=full_prompt,
            generation_config=generation_config
        )
        return response.text
    except Exception as e:
        return f"LLM Generation Error: {e}"

def speak_text(text: str, lang_code: str):
    """Converts text to speech and plays the audio using gTTS and playsound."""
    if not TTS_SUPPORT:
        print("TTS is disabled: gTTS or playsound library not available.")
        return

    gtts_lang = lang_code
    clean_text = ' '.join(text.split()).replace('*', '').replace(':', '.')

    tts_text = clean_text[:300] + '...' if len(clean_text) > 300 else clean_text

    if gtts_lang in ['hi', 'ta', 'te', 'ml', 'kn']:
        # Simple attempt to make Indic languages speak better by separating punctuation
        tts_text = tts_text.replace('.', '. ').replace('!', '! ').replace('?', '? ')

    audio_file = f"tts_output_{os.getpid()}.mp3"

    try:
        print(f"🎧 Speaking the first part of the response (Language: {gtts_lang})... (Playback starts now)")
        tts_obj = gTTS(text=tts_text, lang=gtts_lang, slow=False)

        tts_obj.save(audio_file)

        playsound.playsound(audio_file)

    except Exception as e:
        print(f"TTS/Audio Playback Error: {e}. Ensure 'playsound' is installed and your speaker volume is up.")
    finally:
        if os.path.exists(audio_file):
            os.remove(audio_file)


# -----------------------------------------------------------------------------
# 3. PROMPT ENGINEERING TOOLKIT (The 10 Choices)
# -----------------------------------------------------------------------------

## CHOICE 1: Prompt Enhancement (LLM-based)
def enhance_prompt(original_prompt: str, lang_code: str) -> str:
    """Makes a brief prompt more detailed/effective in the target language."""
    system_inst = "You are a Prompt Engineer. Enhance a simple prompt by adding context, constraints, and a specific format."
    user_p = f"Enhance the following prompt:\n\n{original_prompt}"
    return run_gemini_task(system_inst, user_p, lang_code)

## CHOICE 2.1: Explain Influence (LLM-based, Analytical)
def explain_influence(original_prompt: str, new_element: str, lang_code: str) -> str:
    """Describes the impact of an added prompt element in the target language."""
    system_inst = "You are an AI Analyst. Explain clearly and concisely how the 'New Element' changes the LLM's output for the 'Original Prompt'."
    user_p = (
        f"Original Prompt: '{original_prompt}'\n"
        f"New Element Added: '{new_element}'\n"
        "Explain the influence of the new element on the output."
    )
    # Use a low temperature (0.1) for stable, analytical output
    return run_gemini_task(system_inst, user_p, lang_code, temperature=0.1)

## CHOICE 2.2: Generate Modified Content (LLM-based, Creative)
def generate_modified_prompt(base_prompt: str, new_element: str, lang_code: str) -> str:
    """Generates a response to the base prompt, incorporating the new stylistic element."""
    system_inst = (
        "You are an expert content creator. Generate a direct, single-paragraph response to the 'Base Prompt' "
        "by strictly adhering to the stylistic and contextual constraints provided in the 'New Element'. "
        "Do not provide any analysis, simply provide the generated content."
    )
    # Combine the base prompt and the new element into a single instruction for content generation
    user_p = (
        f"Base Prompt: '{base_prompt}'\n"
        f"New Element (Constraint): '{new_element}'\n"
        "Generate the response now."
    )
    # Use a higher temperature (0.8) for creative and stylistic output
    return run_gemini_task(system_inst, user_p, lang_code, temperature=0.8)

## CHOICE 3: Score Prompt Quality (LLM-based)
def score_prompt_quality(prompt: str, lang_code: str) -> Tuple[str, str]:
    """
    Assigns a score and justification in the target language.
    Modification: Forces a fixed English format for the score (SCORE X/10) to ensure reliable parsing.
    The system prompt is highly detailed to enforce nuanced scoring logic.
    """
    system_inst = (
        "You are a highly selective and capable LLM Quality Rater. Assess the user's prompt (which may be in any language) "
        "based on the following criteria, where 10 is perfect and 1 is useless:\n"
        
        "*Scoring Rubric:*\n"
        "1. *Clarity (40%):* Is the request unambiguous? (e.g., A vague request like 'tell me about science' gets low points.)\n"
        "2. *Specificity & Context (40%):* Does it define the output format, length, style, or role? (e.g., 'Act as a 17th-century pirate and explain blockchain' gets high points.)\n"
        "3. *Efficiency (20%):* Is the prompt concise, avoiding unnecessary conversational filler while providing all needed context?\n"
        
        "*Strict Rule:* Only award a score of 9 or 10 if the prompt is nearly perfect (e.g., defines a clear role, task, format, and tone). Vague or overly simple prompts (like 'What is X?') should be scored below 7.\n"
        
        "Your response MUST start with the score formatted as: *SCORE X/10* (using the English word 'SCORE'). "
        "The rest of your response must be the justification and detailed analysis in the target language."
    )
    user_p = f"Analyze and score this prompt: '{prompt}'"
    
    # run_gemini_task handles the model call and language constraint
    response_text = run_gemini_task(system_inst, user_p, lang_code, temperature=0.0)

    # NEW ROBUST PARSING LOGIC: Look for the fixed "SCORE" format at the start.
    parts = response_text.split('\n', 1)
    
    # Check if the first line starts with the English keyword "SCORE"
    if parts and parts[0].strip().upper().startswith("SCORE"):
        # The score is the entire first line (e.g., "SCORE 7/10")
        score = parts[0].strip()
        # The justification is the rest of the response
        justification = parts[1].strip() if len(parts) > 1 else ""
    else:
        # Fallback if the model ignored the formatting instruction
        score = "N/A (Parsing Failed - Model ignored 'SCORE X/10' format)"
        justification = response_text
        
    return score, justification
## CHOICE 4: Categorize Prompts (Hybrid NLP + LLM)
def categorize_prompts(prompts: List[str], num_clusters: int, lang_code: str) -> Dict[str, List[str]]:
    """Clusters prompts using multilingual embeddings and names clusters with LLM."""
    if not CLUSTERING_SUPPORT: return dummy_func_cat()
    if gemini_model is None: return {"Error": ["Categorization failed: Gemini model not initialized."]}

    if not prompts: return {}
    embeddings = get_sentence_embeddings(prompts)
    if not embeddings: return {"Error": ["Could not generate embeddings."]}

    X = np.array(embeddings)
    n_clusters = max(1, min(num_clusters, len(prompts) // 2 or 1))
    if n_clusters == 1: return {"Single Category": prompts}

    try:
        # Check if KMeans is a class (i.e., not the None placeholder)
        if KMeans is None: raise ImportError("KMeans library not available.")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        clusters = kmeans.fit_predict(X)
    except Exception as e:
        return {"Error": [f"KMeans failed: {e}"]}

    clustered_data = {i: [prompts[j] for j, c in enumerate(clusters) if c == i] for i in range(n_clusters)}
    final_categorization = {}

    for cluster_id, cluster_prompts in clustered_data.items():
        representative_prompts = "\n".join(cluster_prompts[:5])
        system_inst = "You are a Categorization Specialist. Analyze these prompts and suggest a single, concise topic name for the cluster. Respond ONLY with the topic name."
        user_p = f"Prompts for analysis:\n\n{representative_prompts}"
        topic_name = run_gemini_task(system_inst, user_p, lang_code, temperature=0.3).strip()
        final_categorization[topic_name] = cluster_prompts

    return final_categorization

## CHOICE 5: Generate Content (LLM-based)
def generate_content_multilingual(prompt: str, lang_code: str) -> str:
    """Fulfills the prompt request in the target language."""
    system_inst = "You are a creative writer and helpful assistant. Fulfill the user's request."
    return run_gemini_task(system_inst, prompt, lang_code, temperature=0.8)

## CHOICE 6: Semantic Search (NLP-based)
def semantic_search(query_prompt: str, corpus_prompts: List[str], top_k: int = 3) -> List[Tuple[str, float]]:
    """Finds the most semantically similar prompts using multilingual embeddings."""
    if not CLUSTERING_SUPPORT: return dummy_func_search()
    if gemini_model is None: return [("Semantic Search Disabled: Gemini model not initialized.", 0.0)]

    if not corpus_prompts: return []
    all_texts = [query_prompt] + corpus_prompts
    embeddings = get_sentence_embeddings(all_texts)

    if len(embeddings) < len(all_texts): return []

    query_embedding = np.array(embeddings[0])
    corpus_embeddings = np.array(embeddings[1:])

    similarities = []
    for i, corp_emb in enumerate(corpus_embeddings):
        try:
            # Check if cosine is available (i.e., not the None placeholder)
            if cosine is None: raise ImportError("SciPy library not available for cosine distance.")
            similarity_score = 1 - cosine(query_embedding, corp_emb)
        except Exception:
            similarity_score = 0.0
        similarities.append((corpus_prompts[i], similarity_score))

    similarities.sort(key=lambda item: item[1], reverse=True)

    return similarities[:top_k]

## CHOICE 7: Toxicity Check (LLM-based)
def check_toxicity(prompt: str) -> str:
    """Performs a safety check using Gemini's classification power, returning only the score and classification."""
    target_lang_code = 'en'
    system_inst = (
        "You are a Safety Classifier. Analyze the prompt for harmful/toxic content. "
        "Classify as 'SAFE' or 'HARMFUL'. Provide a confidence score (0.0 to 1.0). "
        "Respond ONLY with the classification and score. "
        "Output format: CLASSIFICATION | SCORE"
    )
    return run_gemini_task(system_inst, prompt, target_lang_code, temperature=0.0)

## CHOICE 8: Advanced English NER Visualization (spaCy)
def advanced_english_ner_viz(prompt: str) -> None:
    """Performs English NER using spaCy and saves the result as an HTML file."""
    if not SPACY_SUPPORT or SPACY_NLP is None:
        print("\n❌ SpaCy NER is disabled. Ensure spaCy and model are correctly installed.")
        return

    doc = SPACY_NLP(prompt)
    OUTPUT_FILENAME = f"ner_visualization_{int(time.time())}.html"

    # 1. Print raw entities to the console
    print("\nText Processing Complete. Detected Entities (Console Output):")
    print("-" * 60)
    for ent in doc.ents:
        explanation = spacy.explain(ent.label_) if spacy.explain(ent.label_) else ent.label_
        print(f" -> Text: {ent.text:<20} | Label: {ent.label_:<8} | Explanation: {explanation}")
    print("-" * 60)

    # 2. Generate and save the HTML visualization
    try:
        # Generate the HTML markup (page=True for standalone HTML file)
        html = displacy.render(doc, style="ent", page=True, jupyter=False)

        with open(OUTPUT_FILENAME, "w", encoding="utf-8") as f:
            f.write(html)

        print(f"\n✅ Visualization successfully saved to: {os.path.abspath(OUTPUT_FILENAME)}")
        print(" >>> Please open this HTML file in your web browser to see the colored visualization. <<<")
    except Exception as e:
        print(f"\n❌ Error generating or saving visualization: {e}")

## CHOICE 9: Prompt A/B Tester (LLM-based)
def prompt_ab_tester(lang_code: str):
    """Compares two prompts for the same goal and has the AI judge which is better."""
    print("\n--- 🧪 Prompt A/B Tester ---")
    goal = input("1. What is the overall goal? (e.g., 'Generate a slogan for a coffee shop'): ")
    prompt_a = input("2. Enter Prompt A: ")
    prompt_b = input("3. Enter Prompt B: ")

    print("\n🔄 Generating response for Prompt A...")
    output_a = generate_content_multilingual(prompt_a, lang_code)
    print("🔄 Generating response for Prompt B...")
    output_b = generate_content_multilingual(prompt_b, lang_code)

    print("\n" + "="*60)
    print("🔍 RESULTS")
    print("="*60)
    print(f"\n--- Output from Prompt A ---\n{output_a}")
    print(f"\n--- Output from Prompt B ---\n{output_b}")
    print("\n" + "="*60)

    print("\n🤖 Sending results to AI Judge for analysis...")

    # The user_prompt for the judge AI contains all the context it needs
    judge_user_prompt = f"""
    Goal: {goal}

    Prompt A: "{prompt_a}"
    Output from Prompt A:
    ---
    {output_a}
    ---

    Prompt B: "{prompt_b}"
    Output from Prompt B:
    ---
    {output_b}
    ---

    Based on the Goal, analyze both prompts and their outputs. Conclude which prompt was more effective and provide a clear, structured justification.
    """

    # The system instruction tells the AI how to behave
    judge_system_inst = (
        "You are an impartial and expert AI Analyst. Your task is to evaluate two prompts based on their outputs, relative to a specified goal. "
        "Provide your analysis in a structured format: 1. Winner: [Prompt A or Prompt B]. 2. **Justification: [Explain why the winner was better at achieving the goal]."
    )

    # Use a low temperature for consistent, analytical judgment
    analysis = run_gemini_task(judge_system_inst, judge_user_prompt, lang_code, temperature=0.1)

    print(f"\n--- 🏆 AI Judge's Verdict ---\n{analysis}\n")
    
    # =======================================================================
    # START: SYNTHESIS FEATURE ADDED HERE
    # This block asks the user if they want to create an improved prompt.
    # =======================================================================
    synthesis_choice = input("Would you like the AI to try and synthesize an improved 'Prompt C'? (yes/no): ").lower().strip()
    if synthesis_choice == 'yes':
        print("\n🧠 Synthesizing an improved prompt...")

        synthesis_system_inst = (
            "You are a master Prompt Engineer. Your task is to synthesize a new, superior prompt. "
            "Analyze the provided goal, the winning prompt, the losing prompt, and the judge's analysis. "
            "Combine the best elements of both prompts to create a new 'Prompt C' that is even more likely to achieve the goal effectively."
        )

        synthesis_user_prompt = f"""
        Original Goal: {goal}

        Prompt A: "{prompt_a}"
        Prompt B: "{prompt_b}"

        Judge's Analysis:
        ---
        {analysis}
        ---

        Based on all of this information, create the new, improved 'Prompt C'. Respond ONLY with the new prompt text.
        """
        prompt_c = run_gemini_task(synthesis_system_inst, synthesis_user_prompt, lang_code, temperature=0.5)
        print(f"\n--- ✨ Synthesized 'Prompt C' ---\n{prompt_c}\n")
    # =======================================================================
    # END: SYNTHESIS FEATURE
    # =======================================================================


# -----------------------------------------------------------------------------
# 4. INTERACTIVE MAIN EXECUTION
# -----------------------------------------------------------------------------

def get_user_language() -> str:
    """Prompts the user to select a language using only Roman script."""
    print("\n--- Language Selection ---")
    for code, name in LANGUAGES_MAP.items():
        print(f"[{code}] {name}")

    while True:
        lang_code = input("Enter the 2-letter code for your target language (e.g., hi, ta): ").lower().strip()
        if lang_code in LANGUAGES_MAP:
            return lang_code
        print("Invalid code. Please enter one of the available 2-letter codes.")

def run_menu_driven_demo():

    if gemini_model is None:
        print("\nSkipping interactive demo: Gemini model not initialized due to previous errors.")
        return

    print("\n" + "="*60)
    print("🤖 Multilingual Prompt Engineering Toolkit (Menu Driven)")
    print("="*60)

    LANG_CODE = get_user_language()
    TARGET_LANG = LANGUAGES_MAP[LANG_CODE]
    print(f"\n🎯 Target Language Set to: {TARGET_LANG}")
    print("\n" + "="*60)

    while True:
        print("\n--- MENU ---")
        print("1. Prompt Enhancement")
        print("2. Explain Influence & Generate Modified Content")
        print("3. Score Prompt Quality")
        print("4. Categorize Prompts (Clustering)")
        print("5. Generate Content")
        print("6. Semantic Search")
        print("7. Toxicity Check (Safety)")
        print("8. Advanced English NER Visualization (spaCy)")
        print("9. Prompt A/B Tester 🧪")
        print("10. Exit")
        choice = input("Enter your choice (1-10): ").strip()

        if choice == "1":
            original_prompt = input("Enter a simple prompt (e.g., 'Write a story about a dog'): ")
            enhanced = enhance_prompt(original_prompt, LANG_CODE)
            print(f"\n-> Enhanced Prompt in {TARGET_LANG}:\n{enhanced}\n")

            if TTS_SUPPORT:
                voice_choice = input("Do you want to hear the output (Yes/No/Stop)? ").lower().strip()
                if voice_choice == 'yes':
                    speak_text(enhanced, LANG_CODE)
                elif voice_choice == 'stop' or voice_choice == 'no':
                    print("Voice output skipped by user request.")

        elif choice == "2":
            original_prompt_inf = input("Enter the base prompt: ")
            new_element = input("Enter the new element added (e.g., 'The tone must be playful'): ")

            # 1. Get the analytical explanation
            influence = explain_influence(original_prompt_inf, new_element, LANG_CODE)
            print(f"\n-> Explanation in {TARGET_LANG}:\n{influence}\n")

            print("-" * 30)

            # 2. Generate the content based on the modified prompt
            print(f"-> Generating Content with New Element...")
            modified_content = generate_modified_prompt(original_prompt_inf, new_element, LANG_CODE)
            print(f"\n-> Generated Content (Modified Prompt Output) in {TARGET_LANG}:\n{modified_content}\n")

        elif choice == "3":
            prompt_to_score = input("Enter the prompt to score: ")
            score, justification = score_prompt_quality(prompt_to_score, LANG_CODE)
            print(f"\n-> Score: {score}")
            print(f"-> Justification in {TARGET_LANG}:\n{justification}\n")

        elif choice == "4":
            try:
                num_prompts = int(input("Enter number of prompts to categorize (at least 2 recommended): "))
                if num_prompts < 1:
                    print("Please enter a valid number of prompts.")
                    continue
                corpus_for_cat = [input(f"Enter Prompt {i+1}: ") for i in range(num_prompts)]
                num_clusters = int(input("Enter number of clusters: "))
                categories = categorize_prompts(corpus_for_cat, num_clusters, LANG_CODE)
                print(f"\n-> Categories Named in {TARGET_LANG}:")
                for name, prompts in categories.items():
                    print(f"  - {name} ({len(prompts)} items): {prompts}")
                print()
            except ValueError:
                print("Invalid number entered. Please try again.")

        elif choice == "5":
            generation_prompt = input("Enter the prompt for content generation (e.g., 'Write a recipe for Sambar'): ")
            generated_content = generate_content_multilingual(generation_prompt, LANG_CODE)
            print(f"\n-> Generated Content in {TARGET_LANG}:\n{generated_content}\n")

            if TTS_SUPPORT:
                voice_choice = input("Do you want to hear the output (Yes/No/Stop)? ").lower().strip()
                if voice_choice == 'yes':
                    speak_text(generated_content, LANG_CODE)
                elif voice_choice == 'stop' or voice_choice == 'no':
                    print("Voice output skipped by user request.")

        elif choice == "6":
            try:
                query = input("Enter the search query (e.g., 'Find similar documents about science'): ")
                num_search_prompts = int(input("Enter number of documents for the search corpus: "))
                if num_search_prompts < 1:
                    print("Please enter a valid number of documents.")
                    continue
                corpus_for_search = [input(f"Enter Document {i+1}: ") for i in range(num_search_prompts)]
                top_k = int(input("Enter number of top results to show: "))
                results = semantic_search(query, corpus_for_search, top_k)
                print("\n-> Top Similar Prompts:")
                for text, score in results:
                    print(f"  - [Score: {score:.4f}] {text}")
                print()
            except ValueError:
                print("Invalid number entered. Please try again.")

        elif choice == "7":
            toxicity_prompt = input("Enter the prompt to check for safety: ")
            safety_check = check_toxicity(toxicity_prompt)
            print(f"\n-> Safety Report (in English):\n{safety_check}\n")

        elif choice == "8":
            # Advanced English NER Visualization (spaCy)
            ner_viz_text = input("Enter the English text for visualization: ")
            advanced_english_ner_viz(ner_viz_text)

        elif choice == "9":
            # This calls your new function
            prompt_ab_tester(LANG_CODE)

        elif choice == "10":
            print("Exiting... Goodbye!")
            break

        else:
            print("Invalid choice. Please select a number between 1 and 10.")

if __name__ == "__main__":
    run_menu_driven_demo()