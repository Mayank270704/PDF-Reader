import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="AI Document Intelligence System")

st.title("📄 AI-Powered Document Intelligence System")
st.write(
    "Upload documents and ask natural-language questions. "
    "The system retrieves relevant information using semantic search."
)

# -----------------------
# Upload Section
# -----------------------
st.header("Upload Document")

uploaded_file = st.file_uploader("Choose a .txt file", type=["txt"])

if uploaded_file is not None:
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            "text/plain",
        )
    }

    if st.button("Upload & Index"):
        response = requests.post(f"{API_URL}/upload", files=files)

        if response.status_code == 200:
            st.success("Document uploaded and indexed successfully.")
        else:
            st.error("Failed to upload document.")

# -----------------------
# Query Section
# -----------------------
st.header("Ask a Question")

query = st.text_input("Enter your question")

if st.button("Get Answer"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        response = requests.post(
            f"{API_URL}/query",
            params={"query": query},
        )

        if response.status_code == 200:
            result = response.json()
            st.subheader("Answer")
            st.write(result["answer"])

            if "sources" in result:
                with st.expander("Sources"):
                    for i, src in enumerate(result["sources"], 1):
                        st.write(f"{i}. {src}")
        else:
            st.error("Failed to get answer from backend.")
