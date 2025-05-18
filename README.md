# FMVRAG
# 📊 FMVRAG: Financial Market Volatility Retrieval-Augmented Generation Agent

**FMVRAG** is an intelligent agentic system that combines local document retrieval, real-time financial data APIs, and large language models to help investors understand how foreign exchange rates affect stock market volatility. The system integrates:

- 🔍 Retrieval-Augmented Generation (RAG)
- 🌐 Web search via [Tavily API](https://www.tavily.com/)
- 📈 Volatility data from [yFinance](https://pypi.org/project/yfinance/)
- 🧠 LLMs from [OpenAI](https://platform.openai.com/)
- 📚 Vector database powered by [FAISS](https://github.com/facebookresearch/faiss)

---

## 💡 Problem Background

Our collaborator from JHU SAIS provided a key business problem: many retail investors respond emotionally to FX fluctuations, tariffs, and macroeconomic signals without understanding how these factors truly impact market volatility. Their research examined the relationship between exchange rate movements (e.g., USD/CNY) and index volatility — identifying a need for tools that make such insights accessible to newcomers in equities and derivatives markets.

FMVRAG addresses this challenge by transforming academic and real-time financial data into natural language explanations tailored for less experienced investors.

---

## ⚙️ Features

- ✅ **RAG-powered Q&A** with vector search fallback to real-time web search
- ✅ **USD-to-Any FX Rate Retrieval** using Tavily
- ✅ **Volatility Estimation** from yFinance (with automatic web fallback)
- ✅ **Web Search Summarization** with source URLs
- ✅ **Local Document Retrieval** with FAISS vector store
- ✅ **User-friendly UI** via Streamlit

---

## 🏗️ Project Structure
FMVRAG/
│
├── app.py # Streamlit application
├── requirements.txt # Python dependencies
├── .streamlit/
│ └── secrets.toml # API keys
├── financial_data/
│ └── FinancialVectorDB/ # FAISS vector index
├── docs/
│ └── # Research docs used for retrieval
└── README.md


---

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/ChunjiangLiLeo/fmv-rag-agent.git
cd fmv-rag-agent

### 2. Create a Conda Environment
conda create -n fmv_env python=3.10
conda activate fmv_env
pip install -r requirements.txt
OPENAI_API_KEY = "your-openai-api-key"
TAVILY_API_KEY = "your-tavily-api-key"

###3. Install Dependencies
pip install -r requirements.txt

###4. Set Up API Keys
Create a file at .streamlit/secrets.toml and add:
OPENAI_API_KEY = "your-openai-api-key"
TAVILY_API_KEY = "your-tavily-api-key"

###5 Run the App
streamlit run app.py

📘 How It Works
1。User Input → The user asks a financial question.

2. Local Retrieval → The agent checks relevant research using a FAISS vector DB.

3. Web/API Fallback → If more context is needed, the agent:

4. Gets real-time FX data from Tavily

5. Gets index volatility from Yahoo Finance

6. LLM Response → A GPT-4 model synthesizes all inputs into a human-readable answer.

📈 Volatility Logic
If available, the agent calculates annualized volatility from historical index data:
Volatility = Std(returns) × √252
If API rate limits are hit, the agent searches the web for volatility estimates.

📄 References
Kanas, A. (2000). Exchange Rate Volatility and Its Effect on Stock Market Volatility.

Adjasi, C., Biekpe, N. (2011). Stock Market Returns and Exchange Rate Movements in Africa.

Ng, A. (2000). Volatility Spillover Effects from Japan and the U.S. to Pacific-Basin Markets.

🤝 Contributors
🧠 AI Architecture & Code – Chunjiang Li，Hanzhi Xi, Nanxi Wang, Yijia Jiang

📈 Financial Theory Input – SAIS Collaborator

💻 Stack – Streamlit, FAISS, LangChain, OpenAI, Tavily

🔮 Future Roadmap
Add sentiment analysis on news data

Enable deeper macroeconomic time-series forecasting

Expand to crypto and commodities volatility modeling

License
This project is licensed under the MIT License.

---

Let me know if you'd like me to generate the accompanying `requirements.txt` or `app.py` template too.





