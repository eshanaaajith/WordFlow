🌐 Multilingual Prompt Engineering Toolkit

A menu-driven, multilingual AI prompt toolkit built with Google’s Gemini API, spaCy, and Transformers.
This project helps users enhance, analyze, and test prompts, perform semantic search, NER visualization, and even text-to-speech playback — all within a single Python interface.

🚀 Features
1. Prompt Engineering Tools

Prompt Enhancement: Automatically expand and improve short prompts.

Influence Explanation: Analyze how changes affect prompt outcomes.

Modified Content Generation: Generate text that reflects prompt edits.

Prompt Scoring: Evaluate clarity, specificity, and efficiency (1–10 scale).

Prompt Categorization: Cluster and label similar prompts intelligently.

Prompt A/B Testing: Compare two prompts and generate an improved version.

2. Language Capabilities

Supports multilingual interaction in:

English, Hindi, Tamil, Telugu, Malayalam, Kannada

3. AI and NLP Utilities

Semantic Search using Gemini embeddings.

Advanced English NER Visualization (via spaCy, saved as HTML).

Toxicity Detection to ensure safe prompt usage.

Text-to-Speech (TTS) for multilingual audio responses (via gTTS + playsound).

🧰 Requirements

Install dependencies using:

pip install google-generativeai spacy gTTS playsound transformers scikit-learn scipy numpy


Optional (recommended):

python -m spacy download en_core_web_sm

🔑 Configuration

Open the script and locate:

API_KEY_DIRECT = "YOUR_API_KEY_HERE"


Replace it with your Google Gemini API key:

API_KEY_DIRECT = "AIzaSyXXXXXX..."

🧠 Project Structure
Section	Description
Imports	Handles all essential and optional dependencies safely
Configuration	Sets model parameters, language map, and API setup
Utility Functions	Core LLM, Embedding, NLP, and TTS functions
Prompt Tools	Implements the 10 key prompt-engineering utilities
Interactive Menu	User-friendly CLI for choosing operations
🧩 Menu Options
Option	Description
1	Enhance a simple prompt
2	Explain influence of a new prompt element
3	Score a prompt’s quality
4	Categorize prompts into clusters
5	Generate content in any supported language
6	Perform semantic search on prompt corpus
7	Check prompt toxicity
8	Visualize Named Entities (NER) using spaCy
9	Compare two prompts (A/B testing + synthesis)
10	Exit the toolkit
🖥️ How to Run

Run the Python file directly in your terminal:

python multilingual_prompt_toolkit.py


Follow the on-screen menu to:

Choose your target language

Enter prompts or queries

View or listen to AI-generated outputs

🗣️ Example
🤖 Multilingual Prompt Engineering Toolkit

--- Language Selection ---
[en] English
[hi] Hindi
[ta] Tamil
[ml] Malayalam

Enter code: hi
🎯 Target Language Set to: Hindi

--- MENU ---
1. Prompt Enhancement
2. Prompt A/B Tester
...

⚠️ Notes

spaCy NER visualization saves an .html file you can open in any browser.

If any library is missing, the code gracefully disables that feature.

For TTS playback, ensure your system audio works and playsound is installed.

If you're on Windows and encounter KMP_DUPLICATE_LIB_OK errors, they are handled automatically.

🧾 License

This project is released under the MIT License.
You are free to use, modify, and distribute it with attribution.