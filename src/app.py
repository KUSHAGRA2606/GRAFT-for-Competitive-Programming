import streamlit as st
import time
from graft_agent import init_system, GraphState

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="GRAFT Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern look
st.markdown("""
<style>
    .stChatFloatingInputContainer {
        padding-bottom: 2rem;
    }
    .status-box {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        font-family: monospace;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# Model Initialization (Cached)
# ==============================================================================
@st.cache_resource(show_spinner=False)
def load_graft_pipeline():
    """Load the LangGraph pipeline once and cache it across sessions."""
    return init_system()

# ==============================================================================
# Sidebar: System Statistics
# ==============================================================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/10/PyTorch_logo_icon.svg/1200px-PyTorch_logo_icon.svg.png", width=50)
    st.title("GRAFT Engine")
    st.markdown("### Graph RAG for Algorithmic Fine-Tuning")
    
    st.divider()
    
    st.markdown("#### 🧠 Model Architecture")
    st.caption("**Base Model:** Qwen2.5-3B-Instruct")
    st.caption("**Quantization:** 4-bit (bitsandbytes nf4)")
    st.caption("**Adapters:** LoRA Fine-Tuned (Rank 16)")
    st.caption("**Hardware:** PyTorch SDPA Acceleration")
    
    st.divider()
    
    st.markdown("#### 🔍 Retrieval Pipeline")
    st.caption("**Strategy:** Hybrid Graph + Semantic Search")
    st.caption("**Vector DB:** FAISS (338 Communities)")
    st.caption("**Bi-Encoder:** all-MiniLM-L6-v2")
    st.caption("**Cross-Encoder:** ms-marco-MiniLM-L-6-v2")
    st.caption("**Context:** Top 7 Extractive Sentences")
    
    st.divider()
    st.markdown("*(Running entirely on local GPU)*")

# ==============================================================================
# Main Chat Interface
# ==============================================================================
st.title("🧠 GRAFT: Algorithmic Assistant")
st.markdown("Ask me anything about competitive programming, data structures, or algorithms.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm the GRAFT Engine. How can I help you optimize your algorithms today?"}
    ]

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("E.g., Explain Segment Trees with Lazy Propagation..."):
    
    # 1. Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Setup Assistant response
    with st.chat_message("assistant"):
        
        # Load pipeline (will only take time on the very first run)
        with st.spinner("Initializing GRAFT Pipeline (if first run)..."):
            graph = load_graft_pipeline()

        initial_state: GraphState = {
            "query": prompt,
            "community_summaries": [],
            "community_answers": [],
            "global_answer": "",
        }

        # 3. Dynamic State Visibility via st.status
        with st.status("🚀 Initializing Pipeline...", expanded=True) as status:
            try:
                # Stream the LangGraph execution to show node transitions
                for output in graph.stream(initial_state):
                    for node_name, state_update in output.items():
                        
                        if node_name == "receive":
                            status.update(label="📥 Running Receive Node: Parsing query...", state="running")
                            st.write(f"Query logged: `{prompt}`")
                            
                        elif node_name == "retrieve":
                            status.update(label="🔍 Retrieving Graph Communities & Reranking...", state="running")
                            st.write(f"FAISS retrieved Top 5 communities.")
                            st.write(f"Cross-Encoder extracted Top 7 sentences.")
                            
                        elif node_name == "answer":
                            status.update(label="⚙️ Generating Final Answer via Qwen 3B...", state="running")
                            st.write(f"Context successfully injected. Running generation...")
                
                status.update(label="✅ Pipeline Execution Complete", state="complete")
                
                # Extract the final answer from the last state update
                final_answer = state_update.get("global_answer", "Error: No answer generated.")
                
            except Exception as e:
                status.update(label="❌ Pipeline Error", state="error")
                st.error(f"Error executing pipeline: {e}")
                final_answer = "Sorry, the pipeline encountered an error."

        # 4. Display final output
        st.markdown(final_answer)
        
        # Save assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": final_answer})
