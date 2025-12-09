import streamlit as st
from google import genai
from google.genai import types
from streamlit_mic_recorder import speech_to_text

# --- 1. CONFIGURATION AND SAFETY ---

SYSTEM_INSTRUCTION = """
You are a helpful, friendly, and **strictly non-diagnostic** Healthcare Companion AI.
Your purpose is to provide general, educational, and organizational health information.

RULES TO STRICTLY FOLLOW:
1.  **NEVER** provide a medical diagnosis, personalized treatment plan, or specific medical advice.
2.  **ALWAYS** preface your response with a strong safety disclaimer.
3.  **DO** encourage the user to consult a qualified healthcare professional (doctor, nurse, or pharmacist) for any symptoms or medical concerns.
4.  **DO** offer reliable, factual, general health information.
5.  **LANGUAGE:** Always output your response in the language requested by the user.
"""

MODEL_NAME = 'gemini-2.5-flash'
APP_TITLE = "🩺 Contextual Health Companion (Text Reader)"

# --- CONFIGURATION CONSTANTS ---
TRIGGER_KEYWORDS = ["symptom", "constipation", "pain", "fever", "headache", "cold"]
AGE_RANGES = ["0-12", "13-17", "18-45", "46-65", "65+"]
GENDER_OPTIONS = ["Male", "Female", "Prefer Not to Say"]

# Language Mapping (For Text Output)
LANGUAGE_MAP = {
    'English (Default)': 'English',
    'Kannada (ಕನ್ನಡ)': 'Kannada',
    'Hindi (हिन्दी)': 'Hindi',
    'Telugu (తెలుగు)': 'Telugu'
}
# -----------------------------

# --- STATE MANAGEMENT ---
if 'asking_for_details' not in st.session_state:
    st.session_state.asking_for_details = False 

if 'user_details' not in st.session_state:
    st.session_state.user_details = {} 

if 'current_language' not in st.session_state:
    st.session_state.current_language = 'English' 

if 'show_prescription_form' not in st.session_state:
    st.session_state.show_prescription_form = False
# -----------------------------

# --- 2. INITIALIZATION FUNCTIONS ---

def get_gemini_client():
    """Initializes, stores, and returns the persistent Gemini Client."""
    if 'gemini_client' in st.session_state:
        return st.session_state['gemini_client']

    if "GEMINI_API_KEY" not in st.secrets:
        st.error("❌ API Key not found. Please set your GEMINI_API_KEY in `.streamlit/secrets.toml`.")
        return None
        
    try:
        client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
        st.session_state['gemini_client'] = client 
        return client
    except Exception as e:
        st.error(f"❌ Error initializing Gemini Client: {e}")
        return None

def reset_chat():
    """Resets the chat session state."""
    client = get_gemini_client() 
    if not client:
        return

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION
    )
    
    st.session_state['gemini_chat'] = client.chats.create(model=MODEL_NAME, config=config)
    
    st.session_state.messages = [{"role": "assistant", "content": 
        "**Welcome!** I am your Healthcare Companion. I can provide health information in English, Kannada, Hindi, or Telugu."}]
    st.session_state.asking_for_details = False 
    st.session_state.user_details = {} 
    st.session_state.show_prescription_form = False
    
    st.rerun() 

# --- HELPER FUNCTION FOR AI RESPONSE (TEXT ONLY) ---

def handle_final_response(base_prompt, is_medicine_request=False):
    """
    Handles the API call and streams the text response.
    Appends language instruction to the prompt.
    """
    target_lang = st.session_state.current_language
    
    # Construct the final prompt with language instruction
    if is_medicine_request:
        final_prompt = f"{base_prompt}\n\nPlease provide this information in **{target_lang}** language."
    else:
        final_prompt = f"{base_prompt}\n\n(Respond in {target_lang} language)"

    # Append user prompt to history
    # For medicine requests, we display a clear label
    display_content = base_prompt if not is_medicine_request else f"Requesting info for medicine: {base_prompt}"
    st.session_state.messages.append({"role": "user", "content": display_content})
    
    full_response = ""
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        if 'gemini_chat' in st.session_state:
            try:
                response_stream = st.session_state['gemini_chat'].send_message_stream(final_prompt) 
                
                for chunk in response_stream:
                    full_response += chunk.text
                    message_placeholder.markdown(full_response + "▌") 

                message_placeholder.markdown(full_response)
                
            except Exception as e:
                full_response = f"An error occurred: {e}"
                message_placeholder.markdown(full_response)
        else:
            full_response = "Error: Chat not initialized."
            message_placeholder.markdown(full_response)

    # Log assistant response to history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": full_response
    })

# --- HELPER FOR CONTEXT FORM SUBMISSION ---

def handle_context_form_submit(user_gender, user_age, user_weight):
    st.session_state.user_details['gender'] = user_gender
    st.session_state.user_details['age'] = user_age
    st.session_state.user_details['weight'] = user_weight
    st.session_state.asking_for_details = False 

    # Find the original symptom
    original_symptom = next(
        (msg['content'] for msg in reversed(st.session_state.messages) 
         if msg['role'] == 'user' and not msg['content'].startswith('Requesting info')),
        "General health inquiry."
    )
    
    prompt = (
        f"Original request: {original_symptom}\n"
        f"User Details - Gender: {user_gender}, Age: {user_age}, Weight: {user_weight}\n"
        "Provide general, educational health information based on this context."
    )

    handle_final_response(prompt)
    st.rerun()

# --- 3. STREAMLIT APP UI ---

st.set_page_config(page_title=APP_TITLE, page_icon="🩺", layout="wide")
st.title(APP_TITLE)

if 'gemini_chat' not in st.session_state:
    reset_chat()
    st.rerun() 

# --- SIDEBAR CONTROLS ---

with st.sidebar:
    st.header("⚙️ Settings")
    
    # 1. LANGUAGE SELECTOR (For Reading)
    st.subheader("Select Reading Language")
    selected_lang_key = st.selectbox(
        "Choose the language for the answer:",
        options=list(LANGUAGE_MAP.keys()),
        index=0
    )
    # Update state immediately
    st.session_state.current_language = LANGUAGE_MAP[selected_lang_key]
    
    st.markdown("---")
    
    # 2. MEDICINE INFO BUTTON
    if st.button("💊 Get Medicine Info"):
        st.session_state.show_prescription_form = not st.session_state.show_prescription_form

    st.markdown("---")
    
    st.button("Clear Chat History", on_click=reset_chat, type="primary")
    
    if st.session_state.user_details:
        st.markdown("---")
        st.caption("Context:")
        for k, v in st.session_state.user_details.items():
            st.caption(f"{k}: {v}")


# --- MAIN CHAT AREA ---

# Safety Disclaimer
with st.container(border=True):
    st.markdown("""
    <div style="padding: 5px;">
    <h4 style="color: #FF4B4B; margin-top: 0;">⚠️ SAFETY FIRST</h4>
    <p>I provide general information only. <b>I am not a doctor.</b> Always consult a professional.</p>
    </div>
    """, unsafe_allow_html=True)

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- INTERACTIVE FORMS ---

# 1. MEDICINE INFORMATION FORM
if st.session_state.show_prescription_form:
    with st.form("medicine_info_form"):
        st.subheader(f"💊 Medicine Information ({st.session_state.current_language})")
        st.info("Enter the medicine name below to understand its usage and associated symptoms.")
        
        medicine_name = st.text_input("Enter Medicine Name (e.g., Dolo 650):")
        
        med_submitted = st.form_submit_button("Get Information")

        if med_submitted and medicine_name:
            # Construct a specific prompt for medicine info
            med_prompt = (
                f"Please explain the general usage, purpose, and common symptoms treated by the medicine: '{medicine_name}'. "
                f"Provide a clear note on when it is typically used."
            )
            # Pass is_medicine_request=True to format it correctly
            handle_final_response(med_prompt, is_medicine_request=True)
st.session_state.show_prescription_form = False
st.rerun()

# 2. CONTEXT DETAILS FORM
if st.session_state.asking_for_details:
    with st.form("context_form"):
        st.info("Please provide details for better accuracy:")
        gender = st.radio("Gender", GENDER_OPTIONS, horizontal=True)
        age = st.selectbox("Age Range", AGE_RANGES)
        weight = st.number_input("Weight (kg)", 1, 300, 70)
        
        if st.form_submit_button("Submit"):
            handle_context_form_submit(gender, age, weight)

# --- MAIN INPUT (Voice & Text) ---

# Only show inputs if forms are closed
if not st.session_state.asking_for_details and not st.session_state.show_prescription_form:
    st.markdown("---")
    col1, col2 = st.columns([1, 4])
    
    with col1:
        voice_text = speech_to_text(
            language='en', 
            start_prompt="🎤 Speak", 
            stop_prompt="🛑 Stop",
            just_once=True,
            key='voice_input'
        )
    
    with col2:
        text_input = st.chat_input("Ask about symptoms...")

    user_input = voice_text or text_input

    if user_input:
        # Check for trigger keywords (Symptom flow)
        if any(k in user_input.lower() for k in TRIGGER_KEYWORDS):
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.asking_for_details = True
            with st.chat_message("assistant"):
                msg = "**Context Required:** Please fill the form above."
                st.session_state.messages.append({"role": "assistant", "content": msg})
                st.markdown(msg)
            st.rerun()
        else:
            # Standard Question flow
            handle_final_response(user_input)

            st.rerun()
