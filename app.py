import streamlit as st
import tempfile
import os

from rag import (
    extract_text_from_pdf,
    create_chunks,
    store_chunks,
    ask_question
)


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Student Doubt-Solving Assistant",
    page_icon="📚",
    layout="wide"
)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "document_processed" not in st.session_state:
    st.session_state.document_processed = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "document_name" not in st.session_state:
    st.session_state.document_name = ""


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("📚 Student Doubt-Solving Assistant")

st.write(
    "Upload a learning PDF and ask questions based on its content."
)

st.divider()


# --------------------------------------------------
# SIDEBAR - PDF UPLOAD
# --------------------------------------------------

with st.sidebar:

    st.header("📄 Learning Material")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"]
    )

    if uploaded_file:

        st.success(
            f"Uploaded: {uploaded_file.name}"
        )

        if st.button(
            "🔄 Process Document",
            use_container_width=True
        ):

            try:

                with st.spinner(
                    "Processing document..."
                ):

                    # Save uploaded PDF temporarily
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".pdf"
                    ) as temp_file:

                        temp_file.write(
                            uploaded_file.getvalue()
                        )

                        temp_path = temp_file.name

                    # Extract text
                    pages = extract_text_from_pdf(
                        temp_path
                    )

                    if not pages:
                        st.error(
                            "Could not extract text from the PDF."
                        )
                        st.stop()

                    # Create chunks
                    chunks = create_chunks(
                        pages
                    )

                    # Store in Pinecone
                    chunk_count = store_chunks(
                        chunks
                    )

                    # Save state
                    st.session_state.document_processed = True
                    st.session_state.document_name = uploaded_file.name
                    st.session_state.chat_history = []

                    st.success(
                        f"Document processed successfully! "
                        f"{chunk_count} chunks stored."
                    )

                    # Remove temporary file
                    os.unlink(temp_path)

            except Exception as e:

                st.error(
                    f"Error processing document: {str(e)}"
                )


# --------------------------------------------------
# MAIN APPLICATION
# --------------------------------------------------

if not st.session_state.document_processed:

    st.info(
        "👈 Upload a PDF from the sidebar and click "
        "**Process Document** to get started."
    )

else:

    st.success(
        f"📖 Active document: "
        f"**{st.session_state.document_name}**"
    )

    st.divider()

    # --------------------------------------------------
    # CHAT HISTORY
    # --------------------------------------------------

    for message in st.session_state.chat_history:

        if message["role"] == "user":

            with st.chat_message("user"):
                st.write(message["content"])

        else:

            with st.chat_message("assistant"):

                st.write(message["content"])

                # Display sources
                if message.get("sources"):

                    with st.expander(
                        "📚 View Sources"
                    ):

                        for i, source in enumerate(
                            message["sources"],
                            start=1
                        ):

                            st.markdown(
                                f"**Source {i} — Page "
                                f"{source['page']}**"
                            )

                            st.write(
                                source["text"]
                            )

                            st.caption(
                                f"Similarity score: "
                                f"{source['score']:.4f}"
                            )

                            st.divider()


    # --------------------------------------------------
    # QUESTION INPUT
    # --------------------------------------------------

    question = st.chat_input(
        "Ask a question about the uploaded document..."
    )

    if question:

        # Display user question
        with st.chat_message("user"):
            st.write(question)

        # Save user question
        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": question
            }
        )

        # Generate answer
        with st.chat_message("assistant"):

            with st.spinner(
                "Searching the document..."
            ):

                try:

                    result = ask_question(
                        question
                    )

                    answer = result["answer"]
                    sources = result["sources"]

                    st.write(answer)

                    # Display sources
                    if sources:

                        with st.expander(
                            "📚 View Sources"
                        ):

                            for i, source in enumerate(
                                sources,
                                start=1
                            ):

                                st.markdown(
                                    f"**Source {i} — Page "
                                    f"{source['page']}**"
                                )

                                st.write(
                                    source["text"]
                                )

                                st.caption(
                                    f"Similarity score: "
                                    f"{source['score']:.4f}"
                                )

                                st.divider()

                    # Save assistant response
                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        }
                    )

                except Exception as e:

                    error_message = (
                        f"Something went wrong: {str(e)}"
                    )

                    st.error(error_message)

                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": error_message,
                            "sources": []
                        }
                    )


# --------------------------------------------------
# CLEAR CHAT
# --------------------------------------------------

if st.session_state.document_processed:

    st.sidebar.divider()

    if st.sidebar.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.chat_history = []

        st.rerun()