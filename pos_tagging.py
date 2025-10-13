import nltk
from nltk.tokenize import word_tokenize
import spacy
import matplotlib.pyplot as plt
from collections import Counter
from spacy import displacy # For visualizing spaCy results

# --- NLTK Setup (Download necessary data if not already present) ---
# Ensure nltk is installed: pip install nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError: # Corrected exception type
    nltk.download('punkt')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError: # Corrected exception type
    # It seems NLTK might be looking for the English-specific tagger
    try:
        nltk.download('averaged_perceptron_tagger_eng')
    except LookupError:
        print("Could not download 'averaged_perceptron_tagger' or 'averaged_perceptron_tagger_eng'. NLTK POS tagging may fail.")


# --- spaCy Setup (Load the English model) ---
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    # This will typically run only once if the model isn't found
    #!python -m spacy download en_core_web_sm
    nlp = spacy.load("en_core_web_sm")

print("\n--- NLP Libraries Initialized ---")
print("Ready for Part-of-Speech (POS) Tagging and Visualization!")

# --- Function to get user input ---
def get_user_text():
    """Prompts the user to enter text for analysis."""
    print("\n-----------------------------------------------------")
    user_input = input("Please enter the text you want to analyze (or type 'quit' to exit): \n")
    print("-----------------------------------------------------\n")
    return user_input

# --- Main Loop for User Interaction ---
while True:
    user_text = get_user_text()

    if user_text.lower() == 'quit':
        print("Exiting program. Goodbye!")
        break

    if not user_text.strip():
        print("Input cannot be empty. Please try again.")
        continue

    # --- NLTK POS Tagging ---
    print("\n--- NLTK Results ---")
    try:
        nltk_tokens = word_tokenize(user_text)
        nltk_tagged_words = nltk.pos_tag(nltk_tokens)
        print("Tokenized and Tagged Words (NLTK):")
        for word, tag in nltk_tagged_words:
            print(f"  {word:<15} -> {tag}")
    except LookupError:
        print("NLTK POS tagging failed. Required data not found.")


    # --- spaCy POS Tagging ---
    print("\n--- spaCy Results ---")
    try:
        doc = nlp(user_text)
        print("Tokenized and Tagged Words (spaCy - Universal & Fine-grained):")
        for token in doc:
            explanation = spacy.explain(token.tag_) if spacy.explain(token.tag_) else "N/A"
            print(f"  {token.text:<15} -> POS: {token.pos_:<10} | Tag: {token.tag_:<10} | Desc: {explanation}")
    except Exception as e:
        print(f"spaCy processing failed: {e}")


    # --- Visualization 1: Bar Chart of POS Tag Frequencies (using spaCy's Universal Tags) ---
    print("\n--- Generating POS Tag Frequency Bar Chart ---")
    try:
        pos_tags = [token.pos_ for token in doc]
        tag_counts = Counter(pos_tags)

        # Sort tags by frequency for better visualization
        sorted_tag_counts = dict(sorted(tag_counts.items(), key=lambda item: item[1], reverse=True))

        labels = list(sorted_tag_counts.keys())
        counts = list(sorted_tag_counts.values())

        plt.figure(figsize=(10, 6))
        plt.bar(labels, counts, color='skyblue')
        plt.xlabel('Part-of-Speech Tag (Universal)')
        plt.ylabel('Frequency')
        plt.title('Frequency of POS Tags in the Text (spaCy)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        print("--- Bar Chart Displayed ---")
    except Exception as e:
         print(f"Error generating POS frequency chart: {e}")


    # --- Visualization 2: spaCy's DisplaCy Visualizer ---
    print("\n--- Generating spaCy DisplaCy Visualization ---")
    print("This will open an interactive visualization in your browser or display in an IPython/Jupyter environment.")
    # displacy.render can output directly to a browser (serve=True) or to an IPython environment
    # For a simple script, we'll try to render it in a browser or as HTML if in a non-Jupyter env
    try:
        # If in a Jupyter/IPython environment, this will render directly
        displacy.render(doc, style="dep", jupyter=True)
    except Exception:
        # Fallback for non-Jupyter environments to open in browser
        # You might need to adjust this depending on your environment
        print("\nAttempting to open displaCy in a new browser tab...")
        displacy.serve(doc, style="dep", auto_select_port=True)
        print("Check your browser for the displaCy visualization (usually on http://127.0.0.1:5000)")
    print("--- DisplaCy Visualization Displayed/Served ---")
