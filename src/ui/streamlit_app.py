import json
from pathlib import Path

import requests
import sseclient
import streamlit as st
from htbuilder import div, styles
from htbuilder.units import rem

API_URL = "http://127.0.0.1:8001/chat-stream"

UPLOAD_DIR = Path("study_materials")
UPLOAD_DIR.mkdir(exist_ok=True)

OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="Study Buddy", page_icon="✨", layout="wide")

# -----------------------------------------------------------------------------
# Constants and configuration

SUGGESTIONS = {
    ":blue[:material/search:] Research a topic for me": (
        "Can you research the topic of neural networks and give me a comprehensive overview?"
    ),
    ":green[:material/notes:] Generate study notes": (
        "Research photosynthesis and generate structured study notes I can use to prepare for my exam."
    ),
    ":violet[:material/calendar_month:] What's on my exam schedule?": (
        "What exams do I have coming up? Can you show me my exam schedule?"
    ),
    ":orange[:material/event_note:] Create a study plan": (
        "Can you create a study plan for my upcoming Internet of Things exam based on my exam schedule?"
    ),
    ":gray[:material/folder:] Manage my document database": (
        "List all documents currently in the database."
    ),
    ":red[:material/tips_and_updates:] How can you help me study?": (
        "What can you do to help me study? Walk me through all your capabilities."
    ),
}

# -----------------------------------------------------------------------------
# Session state init

if "messages" not in st.session_state:
    st.session_state.messages = []

if "is_busy" not in st.session_state:
    st.session_state.is_busy = False

if "pending_message" not in st.session_state:
    st.session_state.pending_message = None

if "auto_user_message" not in st.session_state:
    st.session_state.auto_user_message = None

# -----------------------------------------------------------------------------
# UI components


@st.dialog("About Study Buddy")
def show_disclaimer_dialog():
    st.caption(
        """
        👋 Hi! I'm your **Study Buddy** — a multi-agent AI system built to supercharge your studying.\n\n

        **Here's what I can do:**\n\n

        - 🔍 **Research** — I can search the web and gather information on any topic you need to study\n
        - 📝 **Study Notes** — I'll research a topic and generate clean, structured notes you can review\n
        - 📅 **Exam Schedule** — Ask me about your upcoming exams and I'll pull up your schedule\n
        - 🗓️ **Study Plan Generation** — I can create personalized study plans based on your exam schedule and deadlines\n
        - 🧠 **Q&A** — Ask me anything about your subjects and I'll give you a thorough answer\n
        - 🗂️ **Document Database Management** — You can list documents, ingest new files, delete documents, or reset the database\n

        Under the hood, specialized agents work together to handle each task — so you always get the right tool for the job.
    """
    )

@st.dialog("Uploaded Documents Info")
def docs_info_dialog():
    st.caption(
        "Deleting files here only removes them from the study_materials folder. "
        "Reset the database to fully remove embedded documents."
    )

# -----------------------------------------------------------------------------
# Sidebar

with st.sidebar:
    st.title("📚 Study Materials")

    # Initialize uploader key
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    uploaded_files = st.file_uploader(
        "Upload files",
        accept_multiple_files=True,
        type=["pdf", "docx", "txt", "html", "pptx"],
        key=f"file_uploader_{st.session_state.uploader_key}",
        disabled=st.session_state.is_busy,
    )

    # Auto-save uploaded files
    if uploaded_files:
        newly_uploaded = []

        for uploaded_file in uploaded_files:
            save_path = UPLOAD_DIR / uploaded_file.name

            if not save_path.exists():
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                newly_uploaded.append(uploaded_file.name)

        if newly_uploaded:
            st.success(f"Uploaded {len(newly_uploaded)} file(s).")

            # Stage ingestion message and set busy before rerun
            # so the UI re-renders with input disabled immediately
            st.session_state.pending_message = (
                "New documents are added, add them to database."
            )
            st.session_state.is_busy = True

            # Reset uploader completely
            st.session_state.uploader_key += 1

            st.rerun()

    st.divider()

    col1, col2 = st.columns([6, 1])

    with col1:
        st.markdown("### 📄 Uploaded Documents")

    with col2:
        st.button(
            "ℹ️", 
            on_click=docs_info_dialog,
            disabled=st.session_state.is_busy
        )

    study_files = sorted(
        [
            f for f in UPLOAD_DIR.iterdir()
            if f.is_file()
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not study_files:
        st.caption("No uploaded documents.")

    for file_path in study_files:
        col1, col2 = st.columns([6, 1])

        with col1:
            st.caption(f"📄 {file_path.name}")

        with col2:
            if st.button(
                "🗑️",
                key=f"delete_upload_{file_path.name}",
                disabled=st.session_state.is_busy,
            ):
                try:
                    file_path.unlink()

                    # Reset uploader so deleted files can be re-uploaded immediately
                    st.session_state.uploader_key += 1

                    st.rerun()

                except Exception as e:
                    st.error(f"Delete failed: {e}")

    st.divider()

    st.subheader("📥 Generated Outputs")

    output_files = sorted(
        OUTPUTS_DIR.glob("*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not output_files:
        st.caption("No generated outputs yet.")

    for file_path in output_files[:10]:
        try:
            with open(file_path, "rb") as f:
                st.download_button(
                    label=f"⬇️ {file_path.name}",
                    data=f,
                    file_name=file_path.name,
                    key=f"download_{file_path.name}",
                    disabled=st.session_state.is_busy,
                )
        except Exception:
            pass

# -----------------------------------------------------------------------------
# Main header

st.html(div(style=styles(font_size=rem(5), line_height=1))["❉"])

title_row = st.container(horizontal=True, vertical_alignment="bottom")

with title_row:
    st.title("Study Buddy", anchor=False, width="stretch")

# -----------------------------------------------------------------------------
# Initial screen logic

if "context_id" not in st.session_state:
    st.session_state.context_id = None

user_just_asked_initial_question = (
    "initial_question" in st.session_state
    and st.session_state.initial_question
)

user_just_clicked_suggestion = (
    "selected_suggestion" in st.session_state
    and st.session_state.selected_suggestion
)

user_first_interaction = (
    user_just_asked_initial_question
    or user_just_clicked_suggestion
)

has_message_history = len(st.session_state.messages) > 0

# -----------------------------------------------------------------------------
# Empty state screen

if not user_first_interaction and not has_message_history and not st.session_state.pending_message:
    with st.container():
        st.chat_input(
            "Ask a question...",
            key="initial_question",
            disabled=st.session_state.is_busy,
        )

        st.pills(
            label="Examples",
            label_visibility="collapsed",
            options=SUGGESTIONS.keys(),
            key="selected_suggestion",
            disabled=st.session_state.is_busy,
        )

    st.button(
        "&nbsp;:small[:gray[:material/balance: About Study Buddy]]",
        type="tertiary",
        on_click=show_disclaimer_dialog,
    )

    st.stop()

# -----------------------------------------------------------------------------
# Chat input (only rendered when NOT busy, so the widget is truly disabled
# before any processing begins — the key trick to prevent interruptions)

if not st.session_state.is_busy:
    raw_input = st.chat_input("Ask a follow-up...")
else:
    # Render a visually disabled placeholder so layout doesn't shift
    st.chat_input("Working on it...", disabled=True)
    raw_input = None

# -----------------------------------------------------------------------------
# Resolve the next user message to process.
# Priority: pending_message (staged from previous rerun) > raw input > initial question > suggestion

if st.session_state.pending_message:
    # Already staged — pick it up and process below
    user_message = st.session_state.pending_message
    st.session_state.pending_message = None

elif raw_input:
    # New message typed by the user — stage it and rerun with is_busy=True
    # so the input is disabled *before* we start processing
    st.session_state.pending_message = raw_input
    st.session_state.is_busy = True
    st.rerun()

elif user_just_asked_initial_question:
    st.session_state.pending_message = st.session_state.initial_question
    st.session_state.is_busy = True
    st.rerun()

elif user_just_clicked_suggestion:
    st.session_state.pending_message = SUGGESTIONS[st.session_state.selected_suggestion]
    st.session_state.is_busy = True
    st.rerun()

else:
    user_message = None

# -----------------------------------------------------------------------------
# Restart button

with title_row:

    def clear_conversation():
        st.session_state.messages = []
        st.session_state.initial_question = None
        st.session_state.selected_suggestion = None
        st.session_state.pending_message = None
        st.session_state.is_busy = False
        st.session_state.context_id = None

    st.button(
        "Restart",
        icon=":material/refresh:",
        on_click=clear_conversation,
        disabled=st.session_state.is_busy,
    )

# -----------------------------------------------------------------------------
# Render chat history

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -----------------------------------------------------------------------------
# Handle new user message
# At this point is_busy is True and the input widget is already rendered
# as disabled, so the user cannot submit anything new.

if user_message:
    user_message = user_message.replace("$", r"\$")

    with st.chat_message("user"):
        st.text(user_message)

    with st.chat_message("assistant"):
        placeholder = st.empty()

        final_response = ""
        error_occurred = False

        try:
            with st.spinner("Working on it..."):
                response = requests.post(
                    API_URL,
                    json={
                        "message": user_message,
                        "context_id": st.session_state.context_id,
                    },
                    stream=True,
                    timeout=300,
                )

                response.raise_for_status()

                client = sseclient.SSEClient(response)

                for event in client.events():
                    data = json.loads(event.data)

                    content = data.get("content", "")

                    if event.event == "update":
                        placeholder.info(content)

                    elif event.event == "final":
                        placeholder.empty()
                        final_response = content
                        st.session_state.context_id = None  # conversation done, reset

                    elif event.event == "input_required":
                        placeholder.empty()
                        final_response = content
                        st.session_state.context_id = data.get("context_id") # Save context_id so the next message resumes the same thread
 
                    elif event.event == "error":
                        placeholder.empty()
                        st.error(content)
                        st.session_state.context_id = None
                        error_occurred = True
                        break

        except requests.exceptions.Timeout:
            placeholder.empty()
            st.error("The request timed out.")
            error_occurred = True

        except requests.exceptions.ConnectionError:
            placeholder.empty()
            st.error("Could not connect to the backend. Is the server running?")
            error_occurred = True

        except requests.exceptions.HTTPError as e:
            placeholder.empty()
            st.error(f"Server error: {e.response.status_code}")
            error_occurred = True

        except Exception as e:
            placeholder.empty()
            st.error(f"Unexpected error: {e}")
            error_occurred = True

    # Persist messages
    if not error_occurred:
        st.session_state.messages.append(
            {"role": "user", "content": user_message}
        )

        if final_response:
            st.session_state.messages.append(
                {"role": "assistant", "content": final_response}
            )

    # Always unblock the UI when done
    st.session_state.is_busy = False

    # Force clean rerender from session state only
    st.rerun()