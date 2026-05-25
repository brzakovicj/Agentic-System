import json
import streamlit as st
from htbuilder.units import rem
from htbuilder import div, styles
import sseclient
import requests

API_URL = "http://127.0.0.1:8001/chat-stream"

st.set_page_config(page_title="Study Buddy", page_icon="✨")

# -----------------------------------------------------------------------------
# Konstante i konfiguracija

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
    ":cyan[:material/folder:] Manage my document database": (
        "List all documents currently in the database."
    ),
    ":red[:material/tips_and_updates:] How can you help me study?": (
        "What can you do to help me study? Walk me through all your capabilities."
    ),
}

# -----------------------------------------------------------------------------
# UI komponente

@st.dialog("About Study Buddy")
def show_disclaimer_dialog():
    st.caption("""
        👋 Hi! I'm your **Study Buddy** — a multi-agent AI system built to supercharge your studying.\n\n
        **Here's what I can do:**\n\n
        - 🔍 **Research** — I can search the web and gather information on any topic you need to study\n
        - 📝 **Study Notes** — I'll research a topic and generate clean, structured notes you can review\n
        - 📅 **Exam Schedule** — Ask me about your upcoming exams and I'll pull up your schedule\n
        - 🗓️ **Study Plan Generation** — I can create personalized study plans based on your exam schedule and deadlines\n
        - 🧠 **Q&A** — Ask me anything about your subjects and I'll give you a thorough answer\n
        - 🗂️ **Document Database Management** — You can list documents, ingest new files, delete documents, or reset the database\n\n
        Under the hood, specialized agents work together to handle each task — so you always get the right tool for the job. 
    """)


# -----------------------------------------------------------------------------
# Crtanje UI-ja

st.html(div(style=styles(font_size=rem(5), line_height=1))["❉"])

title_row = st.container(horizontal=True, vertical_alignment="bottom")

with title_row:
    st.title("Study Buddy", anchor=False, width="stretch")

# Provera stanja sesije
if "context_id" not in st.session_state:
    st.session_state.context_id = None

user_just_asked_initial_question = (
    "initial_question" in st.session_state and st.session_state.initial_question
)
user_just_clicked_suggestion = (
    "selected_suggestion" in st.session_state and st.session_state.selected_suggestion
)
user_first_interaction = user_just_asked_initial_question or user_just_clicked_suggestion
has_message_history = (
    "messages" in st.session_state and len(st.session_state.messages) > 0
)

# Početni ekran — bez istorije poruka
if not user_first_interaction and not has_message_history:
    st.session_state.messages = []

    with st.container():
        st.chat_input("Ask a question...", key="initial_question")

        st.pills(
            label="Examples",
            label_visibility="collapsed",
            options=SUGGESTIONS.keys(),
            key="selected_suggestion",
        )

    st.button(
        "&nbsp;:small[:gray[:material/balance: About Study Buddy]]",
        type="tertiary",
        on_click=show_disclaimer_dialog,
    )

    st.stop()

# Ekran s istorijom — chat input na dnu
user_message = st.chat_input("Ask a follow-up...")

if not user_message:
    if user_just_asked_initial_question:
        user_message = st.session_state.initial_question
    if user_just_clicked_suggestion:
        user_message = SUGGESTIONS[st.session_state.selected_suggestion]

# Dugme za restart
with title_row:
    def clear_conversation():
        st.session_state.messages = []
        st.session_state.initial_question = None
        st.session_state.selected_suggestion = None
        st.session_state.context_id = None

    st.button(
        "Restart",
        icon=":material/refresh:",
        on_click=clear_conversation,
    )

# Prikaz istorije poruka
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            st.container()

        st.markdown(message["content"])

# Nova poruka korisnika
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
                    timeout=120,  # 2 minuta maks. za duže zadatke
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
                        # Save context_id so the next message resumes the same thread
                        st.session_state.context_id = data.get("context_id")
 
                    elif event.event == "error":
                        placeholder.empty()
                        st.error(content)
                        st.session_state.context_id = None
                        error_occurred = True
                        break
 
        except requests.exceptions.Timeout:
            placeholder.empty()
            st.error("The request timed out. Please try again.")
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
 
        if not error_occurred and final_response:
            st.markdown(final_response)
 
    if not error_occurred:
        st.session_state.messages.append({"role": "user", "content": user_message})
        if final_response:
            st.session_state.messages.append({"role": "assistant", "content": final_response})