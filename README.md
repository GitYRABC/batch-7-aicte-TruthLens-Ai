🔍 TruthLens AI — Real-Time Fact Checker

AI-powered fact-checking app that verifies news, articles, and any text against live web sources — returning a verdict, confidence score, and full reasoning.


🚀 Live Demo
Deploy your own → share.streamlit.io

✨ What It Does
Paste any text or upload a PDF and TruthLens will:

Extract the key factual claims using Meta Llama 3
Search the web for each claim via DuckDuckGo (free, no API key)
Verify each claim — Llama 3 reasons over the search results
Output a final verdict with a confidence score and detailed reasoning


📊 Output
OutputDescription✅ / ❌ / ⚠️ VerdictTRUE / FALSE / PARTIALLY TRUE / UNVERIFIABLE📊 Confidence Score0–100 accuracy score📝 ReasoningWhy Llama reached this verdict, citing evidence🔎 Claim BreakdownIndividual verdict + score + reasoning per claim📚 HistoryAll past checks saved in session, exportable as JSON

🧠 How It Works
Your text / PDF
      ↓
① Llama 3 extracts 3–5 key factual claims
      ↓
② DuckDuckGo searches the web for each claim
      ↓
③ Llama 3 reasons over search results per claim
      → Verdict + Score + Reasoning
      ↓
④ Llama 3 aggregates → Final overall verdict

🛠️ Tech Stack
ComponentTechnologyUIStreamlitGenerative AIMeta Llama 3.2-3B via HuggingFace RouterAPI ClientOpenAI SDK (pointed at HF Router)Web SearchDuckDuckGo (free, no key needed)PDF ParsingPyPDF2

📁 Project Structure
truthlens/
├── truthlens_app.py     # Main Streamlit app
├── requirements.txt     # Python dependencies
├── .env                 # Your HF token (never commit this)
├── .gitignore           # Excludes .env from git
└── README.md            # This file

⚙️ Setup & Run Locally
1. Clone the repo
bashgit clone https://github.com/yourusername/truthlens.git
cd truthlens
2. Install dependencies
bashpip install -r requirements.txt
3. Get a HuggingFace token

Go to huggingface.co → Settings → Access Tokens
Create a new token with "Make calls to Inference Providers" permission
It will look like: hf_xxxxxxxxxxxxxxxx

4. Accept Meta's Llama license (one time only)

Visit huggingface.co/meta-llama/Llama-3.2-3B-Instruct
Click Accept on the license gate

5. Create your .env file
HF_TOKEN=hf_xxxxxxxxxxxxxxxx
6. Run the app
bashstreamlit run truthlens_app.py
Open http://localhost:8501 in your browser.

🌐 Deploy to Streamlit Cloud (Free)

Push your project to GitHub (make sure .env is in .gitignore)
Go to share.streamlit.io
Connect your GitHub and select this repo
Set truthlens_app.py as the main file
Go to Settings → Secrets and add:

   HF_TOKEN = "hf_xxxxxxxxxxxxxxxx"

Click Deploy — you'll get a public URL like yourapp.streamlit.app


📦 requirements.txt
streamlit>=1.32.0
openai>=1.0.0
python-dotenv>=1.0.0
PyPDF2>=3.0.0
requests>=2.31.0
duckduckgo-search>=6.0.0

🔒 .gitignore
Make sure this file exists so your token is never exposed:
.env
__pycache__/
*.pyc
.DS_Store

💡 Swapping the Llama Model
To use a different Llama variant, change one line at the top of truthlens_app.py:
python# Current (fast, small)
LLAMA_MODEL = "meta-llama/Llama-3.2-3B-Instruct:hyperbolic"

# Better quality (larger, slower)
LLAMA_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct:hyperbolic"

# Fastest (smallest)
LLAMA_MODEL = "meta-llama/Llama-3.2-1B-Instruct:hyperbolic"
Nothing else in the code needs to change.

⚠️ Limitations

Free tier rate limits — HuggingFace free accounts have request limits; if you hit them, wait a minute and retry
DuckDuckGo search — occasionally rate-limits heavy usage; the app handles this gracefully
Llama 3.2-3B — a smaller model; for higher accuracy on complex claims consider upgrading to the 8B variant
For educational use — not a replacement for professional fact-checking


📄 License
MIT License — free to use, modify, and distribute.

Built with ❤️ using Streamlit · Meta Llama 3 · HuggingFace · DuckDuckGo
