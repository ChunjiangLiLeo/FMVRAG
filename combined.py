import os
import re
import glob
import streamlit as st
import yfinance as yf
from tavily import TavilyClient
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# ✅ API Key Setup
os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"] if "TAVILY_API_KEY" in st.secrets else "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"
os.environ["OPENAI_API_KEY"] = st.secrets.get("OPENAI_API_KEY", "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")

# ✅ UI Initialization
st.set_page_config(page_title="Financial RAG Agent", layout="wide")
st.title("📊 Financial RAG Agent")

# ✅ Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ✅ Local Document Loader & Vectorstore
doc_dir = "/Users/jade/Desktop/Generative AI/FMVRAG/FINDOC"
documents = []
for file in glob.glob(doc_dir + "/*"):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(file)
    elif file.endswith(".txt"):
        loader = TextLoader(file)
    elif file.endswith(".docx"):
        loader = Docx2txtLoader(file)
    else:
        continue
    documents.extend(loader.load())

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunked_docs = splitter.split_documents(documents)

vectorstore_path = "financial_data/FinancialVectorDB"
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
if not os.path.exists(vectorstore_path):
    texts = [doc.page_content for doc in chunked_docs]
    vectorstore = FAISS.from_texts(texts, embedding=embeddings)
    vectorstore.save_local(vectorstore_path, index_name="financial_docs")
else:
    vectorstore = FAISS.load_local(
        folder_path=vectorstore_path,
        embeddings=embeddings,
        index_name="financial_docs",
        allow_dangerous_deserialization=True
    )
retriever = vectorstore.as_retriever()

# ✅ Tools
@tool
def retrieve_documents(query: str) -> str:
    """Retrieve from local vector DB."""
    docs = retriever.get_relevant_documents(query)
    return "\n\n".join([doc.page_content[:300] for doc in docs[:3]])

@tool
def get_latest_fx_rate_tavily(target_currency: str) -> str:
    """Fetch latest USD to target_currency exchange rate."""
    client = TavilyClient()
    query = f"latest USD to {target_currency.upper()} exchange rate"
    result = client.search(query=query, search_depth="basic", include_answer=True)
    return result.get("answer", "Exchange rate not found.")

INDEX_TICKER_MAP = {
    "nasdaq": "^IXIC", "s&p 500": "^GSPC", "dow jones": "^DJI",
    "nikkei 225": "^N225", "ftse 100": "^FTSE", "hang seng": "^HSI", "nasdaq 100": "^NDX"
}

@tool
def get_index_volatility(index_name: str, period_days: int = 30) -> str:
    """Annualized volatility for a stock index."""
    ticker = INDEX_TICKER_MAP.get(index_name.lower().strip())
    if not ticker:
        return f"❌ Unsupported index name: {index_name}"
    try:
        df = yf.Ticker(ticker).history(period=f"{period_days}d")
        if df.empty:
            raise ValueError("No data")
        df['returns'] = df['Close'].pct_change()
        vol = df['returns'].std() * (252 ** 0.5)
        return f"📉 {index_name.upper()} volatility: {vol:.4f}"
    except Exception as e:
        client = TavilyClient()
        fallback_query = f"{index_name} index volatility past {period_days} days"
        result = client.search(query=fallback_query, search_depth="basic", include_answer=True)
        return result.get("answer", f"Tavily fallback failed: {e}")

@tool
def search_web(query: str) -> str:
    """Searches the web using Tavily and returns the results with URLs."""
    client = TavilyClient()
    results = client.search(query=query, max_results=3)
    return "\n\n".join([f"{r['content'][:300]}\n(Source: {r['url']})" for r in results['results']])

# ✅ Agent Definition
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a smart financial assistant. Use tools to retrieve local knowledge, live FX rates, volatility data, and news."),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])
llm = ChatOpenAI(model="gpt-4", temperature=0)
tools = [get_latest_fx_rate_tavily, get_index_volatility, retrieve_documents, search_web]
agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# ✅ Input UI
user_input = st.chat_input("Ask your financial question:")
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.spinner("Analyzing..."):
        result = agent_executor.invoke({"input": user_input})
        response = result["output"]
        st.session_state.chat_history.append({"role": "agent", "content": response})

# ✅ Render Chat History
for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        st.chat_message("user").markdown(chat["content"])
    else:
        st.chat_message("assistant").markdown(chat["content"])