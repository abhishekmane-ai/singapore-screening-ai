import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from PyPDF2 import PdfReader
import os
from phi.agent import Agent
from phi.tools.crawl4ai_tools import Crawl4aiTools
from phi.model.groq import Groq
from phi.tools.googlesearch import GoogleSearch

# Set API Key for Google Generative AI
os.environ["GOOGLE_API_KEY"] = "AIzaSyCLNi5uUWOFIme1x0rL51ZaPmU1hfKlHE8"

# Streamlit Page Config
st.set_page_config(
    page_title="Naipunya Investigator Registry ", 
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
    <style>
        .main { background-color: #f8f9fa; }
        .stMarkdown h1 { color: #1a237e; border-bottom: 3px solid #f50057; padding-bottom: 10px; }
        .stTab [aria-selected='true'] { background-color: #e8f4f8 !important; color: #1a237e !important; font-weight: bold; }
        .stAlert { border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .footer { text-align: center; padding: 20px; color: #666; margin-top: 30px; }
    </style>
""", unsafe_allow_html=True)

# Updated Title Section
st.title("🔍 Naipunya Inspector Registry AI Assistant")
st.markdown("""
    *Your Intelligent Partner for Building Safety Compliance Verification*  
    *Powered by Advanced AI Research from Naipunya AI Labs*
""")
st.markdown("---")

# Tabs
tab_pdf_qa, tab_general, tab_cpib, tab_general_web = st.tabs([
    "📄 PDF Inspector Search", 
    "🏛️ Government Directory", 
    "🕵️ CPIB Verification",
    "🌐 RiskCenter platform Web Search"  # New 4th tab
])

with tab_pdf_qa:
    st.markdown("### 🔍 Query the Inspector Registry PDF")
    st.markdown("""*Upload the latest inspector registry PDF to verify credentials*""")
    
    uploaded_file = st.file_uploader("**STEP 1:** Upload PDF File", type=["pdf"], help="Maximum file size: 50MB", key="pdf_upload")
    user_question = st.text_input("**STEP 2:** Enter Inspector Details to Verify", placeholder="E.g.: 'Find contact information for John Tan with RI number ABC123'", key="pdf_question")

    if uploaded_file and user_question:
        with st.spinner("🔎 Analyzing document... This may take up to 30 seconds"):
            try:
                pdf_reader = PdfReader(uploaded_file)
                raw_text = "".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
                text_chunks = text_splitter.split_text(raw_text)
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
                vector_store.save_local("faiss_index")
                new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
                docs = new_db.similarity_search(user_question)
                
                model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
                prompt_template = PromptTemplate(
                    template="""
                    Analyze this document context: {context}
                    Answer this question: {question}
                    
                    For Singapore building inspectors, extract:
                    - Name
                    - RI Number  
                    - Discipline (A/M&E)
                    - Contact Information
                    
                    If information not found, state clearly.
                    """,
                    input_variables=["context", "question"]
                )

                chain = load_qa_chain(
                    model, 
                    chain_type="stuff", 
                    prompt=prompt_template,
                    document_variable_name="context"  # Explicitly set document variable
                )
                response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
                
                st.success("✅ Verification Complete")
                with st.expander("📋 View Full Details", expanded=True):
                    st.subheader("Query Summary")
                    st.markdown(f"**Search Criteria:**\n{user_question}")
                    st.divider()
                    st.subheader("Registry Findings")
                    st.markdown(response["output_text"].strip())
                    
            except Exception as e:
                st.error(f"❌ Processing Error: {str(e)}")
    else:
        st.info("ℹ️ Please upload a PDF file and enter your verification query to begin")

with tab_general:
    st.markdown("### 🏛️ Government Official Verification")
    
    search_query = st.text_input("**Name to Verify**", placeholder="Enter full name", key="general_query")
    website_url = st.text_input("**Official Website URL**", value="https://www.sgdi.gov.sg/ministries", key="general_url")
    
    if st.button("🔍 Start Verification", use_container_width=True):
        if search_query:
            instructions = [
                """**Role**: You're a Singaporean government directory specialist. Strictly follow these steps:

                1. **Website Navigation**:
                - Access {website_url}
                - Use EXACT search query: "{search_query}" (case-sensitive)

                2. **Search Analysis**:
                - If found: Extract:
                • Full name (exact match)
                • Current position
                • Department/agency
                - IGNORE unrelated results
                
                - If not found: Return "No official record found for [Name] in Singapore government directories"

                """
            ]
            agent = Agent(
                tools=[Crawl4aiTools(max_length=None), GoogleSearch()],
                model=Groq(id="llama-3.3-70b-versatile", api_key="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"),
                instructions=instructions,
                description=("You are a web automation agent designed to search and extract relevant information "
                             "from specified websites. Your task is to identify and report whether a given person's name "
                            "exists on specific websites, such as government directories or award recipient lists."),
                debug_mode=True
            )
            response = agent.run(f"Search for the Singapore Person Name {search_query} on {website_url}").content.strip()
            st.success("✅ Search Complete")
            with st.expander("📄 View Official Record Summary", expanded=True):
                st.markdown(f"**Name Searched:** {search_query}")
                st.markdown(f"**Source:** `{website_url}`")
                st.divider()
                st.markdown(response)
        else:
            st.warning("⚠️ Please enter a name to search")

with tab_cpib:
    st.markdown("### 🕵️ Corruption Case Verification")
    
    search_query_cpib = st.text_input("**Name to Investigate**", placeholder="Enter full name", key="cpib_query")
    
    if st.button("🔍 Start CPIB Check", use_container_width=True):
        if search_query_cpib:
            instructions = [
                "Search for Bad News or Fraud Cases about a Person on a Website:",
                "- Navigate to the provided website (CPIB Press Room) for a given person's name.",
                "- Use the website's search functionality to look for mentions of the person.",
                "Process the Search Results:",
                "- If the person is mentioned in fraud or bad news articles, extract relevant details (e.g., case type, involvement in fraud, etc.).",
                "- If the person is not mentioned, log that no information is available.",
                "Output Results:",
                "- Provide a clear, concise answer indicating whether the person is mentioned in fraud or bad news articles.",
                "- If found, give a brief summary of the case or context (title, role, or case type).",
                "- Avoid providing unnecessary details or unrelated information.",
                "Handle Errors Gracefully:",
                "- If the website cannot be accessed or the search functionality is unavailable, note this briefly in the response."
            ]

            description = (
                "You are a web automation agent designed to search the CPIB Press Room for mentions of a person's name in articles related to fraud or bad news. "
                "Your task is to clearly state if the person is involved in any such cases and summarize any related details, if applicable."
            )
            agent = Agent(
                tools=[Crawl4aiTools(max_length=None)],
                model=Groq(id="llama-3.3-70b-versatile", api_key="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"),
                instructions=instructions,
                description=description,
                debug_mode=True
            )
            response = agent.run(f"Search for fraud cases about {search_query_cpib} in CPIB Press Room").content.strip()
            st.success("✅ Investigation Complete")
            with st.expander("📄 View CPIB Record Summary", expanded=True):
                st.markdown(f"**Subject:** {search_query_cpib}")
                st.divider()
                st.markdown(response)
        else:
            st.warning("⚠️ Please enter a name to investigate")



# Add the new General Web Search tab
with tab_general_web:
    st.markdown("### 🌐 Comprehensive Web Verification")
    
    col1, col2 = st.columns([3, 2])
    with col1:
        search_query_web = st.text_input("**Full Name to Investigate**", 
                                       placeholder="Enter full name to search", 
                                       key="general_web_query")
    with col2:
        website_url_web = "https://riskcenter.dowjones.com/search/simple"
    
    if st.button("🔍 Start Web Investigation", use_container_width=True, key="web_search_btn"):
        if search_query_web and website_url_web:
            with st.spinner("🕵️ Conducting deep web analysis..."):
                try:
                    agent = Agent(
                        tools=[Crawl4aiTools(max_length=None)],
                        model=Groq(id="llama-3.3-70b-versatile", api_key="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"),
                        #instructions=instructions,
                        #description=description,
                        debug_mode=True
                    )
                    response = agent.run(
                        f"Search for the person {search_query_web} on the website {website_url_web}"
                    ).content.strip()
                    
                    st.success("✅ Web Investigation Complete")
                    with st.expander("📄 Full Digital Footprint Report", expanded=True):
                        st.markdown(f"**Subject:** `{search_query_web}`")
                        st.markdown(f"**Sources Scanned:** `{website_url_web}`")
                        st.divider()
                        st.markdown(response)
                        
                except Exception as e:
                    st.error(f"🔴 Investigation Error: {str(e)}")
        else:
            st.warning("⚠️ Please provide both name and website URL for investigation")

st.markdown("---")
st.markdown('<div class="footer">🔒 All searches are confidential </div>', unsafe_allow_html=True)
