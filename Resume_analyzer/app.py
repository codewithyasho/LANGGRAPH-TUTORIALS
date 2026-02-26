from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import streamlit as st
import PyPDF2
import io

load_dotenv()

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Resume Analyzer",
    page_icon="📄",
    layout="centered"
)

st.title("📄 Resume Analyzer")
st.markdown("Upload your resume and get AI-powered insights.")

# -------------------------------
# File Upload + Inputs
# -------------------------------
uploaded_file = st.file_uploader(
    "Choose a resume file (PDF, TXT)", type=["pdf", "txt"]
)

job_role = st.text_input("Enter the job role you are applying for")

analyze_button = st.button("Analyze Resume")


# -------------------------------
# Load LLM (Cached for speed)
# -------------------------------
@st.cache_resource
def load_llm():
    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.2
    )

llm = load_llm()


# -------------------------------
# PDF Text Extractor
# -------------------------------
def pdf_text_extractor(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += (page.extract_text() or "") + "\n"
    return text


def extract_text_from_file(file):
    if file.type == "application/pdf":
        return pdf_text_extractor(io.BytesIO(file.read()))
    return file.read().decode("utf-8")


# -------------------------------
# Resume Analysis
# -------------------------------
if analyze_button:

    if not uploaded_file:
        st.warning("Please upload a resume file.")
        st.stop()

    if not job_role:
        st.warning("Please enter the job role.")
        st.stop()

    try:
        file_content = extract_text_from_file(uploaded_file)

        if not file_content.strip():
            st.error("The uploaded file is empty.")
            st.stop()

        # Preview
        with st.expander("Preview Extracted Resume"):
            st.write(file_content)

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert HR and resume reviewer."),

            ("user", """
            Analyze the following resume for the role of {job_role}.

            Respond in the following structured format:

            ## Overall ATS Score (out of 100)

            ## Overall Impression

            ## Strengths

            ## Weaknesses

            ## Missing Skills for {job_role}

            ## Improvements Needed

            ## Actionable Recommendations

            Resume:
            {file_content}
            """)
        ])

        chain = prompt | llm

        with st.spinner("Analyzing your resume..."):
            placeholder = st.empty()
            output = ""

            try:
                for chunk in chain.stream({
                    "job_role": job_role,
                    "file_content": file_content
                }):
                    if chunk.content:
                        # Handle content that could be str or list
                        content = chunk.content if isinstance(
                            chunk.content, str) else str(chunk.content)
                        output += content
                        placeholder.markdown(output)

                # Show success message after completion
                st.success("✅ Analysis completed successfully!")

            except Exception as stream_error:
                st.error(f"Error during analysis: {str(stream_error)}")
                if output:
                    st.warning("Partial analysis generated before error:")
                    st.markdown(output)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
