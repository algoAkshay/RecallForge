from dotenv import find_dotenv, load_dotenv
import os
import asyncio
import logging
from langchain.agents import create_agent
from langchain_core.caches import InMemoryCache
from langgraph.checkpoint.memory import InMemorySaver
from tools.web_scraper import fetch_sites, visit
from tools.relevance_search import fetch_information

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv(find_dotenv(usecwd=True), override=True)

# Retrieve model identifier from env variables
MODEL = os.environ.get("MODEL", "google_genai:gemini-3.5-flash-lite")

# The default model is served by the Gemini API. Check credentials before
# LangChain attempts provider initialization so startup failures are actionable.
if MODEL.startswith("google_genai:") and not (
    os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
):
    raise RuntimeError(
        "Missing Gemini API credentials. Set GOOGLE_API_KEY in your environment "
        "or .env file. See .env.example for configuration."
    )

# Define tools for web search and content extraction
tools = [fetch_sites, visit, fetch_information]

# Create agent
rag_agent = create_agent(
    model=MODEL,
    tools=tools,
    system_prompt=("You are RecallForge, an intelligent AI research assistant developed by Akshay Kumar. "
        "Evidence is labeled with source IDs such as [S1]. Cite factual claims only with IDs present in "
        "the supplied evidence; never invent IDs or URLs. If evidence is insufficient, say so. "
        "Do not generate a separate bibliography; the application renders authoritative source details."),
    checkpointer=InMemorySaver(),
    cache=InMemoryCache(),
)

# Final synthesis deliberately has no retrieval tools: routing owns evidence acquisition.
synthesis_agent = create_agent(
    model=MODEL,
    tools=[],
    system_prompt=("Synthesize a clear answer from the supplied labeled evidence. Use inline [S#] citation IDs "
        "for factual claims only. Never invent citation IDs or URLs. Do not generate Sources, References, "
        "or Bibliography sections; the application renders the authoritative source list separately. If the supplied evidence is insufficient, return exactly: I couldn't find enough relevant evidence to answer this confidently. "
        "When using "
        "Markdown tables, include explicit pipe separators between every column and a valid separator row; never "
        "concatenate column headings."),
    cache=InMemoryCache(),
)
