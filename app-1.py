import os
import streamlit as st
import base64
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from streamlit_mic_recorder import speech_to_text
from gtts import gTTS  # Import gTTS for text-to-speech
import io
import streamlit.components.v1 as components

# Import the modules
from module_meaning import setup_rag_pipeline_meaning
from module_example import setup_rag_pipeline_example
from module_translation import setup_translation_chain
from module_melum_kooru import setup_melum_kooru_chain
from module_nirapuga import validate_nirappugaa_answers, generate_nirappugaa_exercise
from module_kurippu_eludhuthal import setup_rag_pipeline_kurippu_eludhuthal
from module_karutharithal import validate_karutharithal_answers, generate_karutharithal_exercise
from expand_further import setup_expand_further_chain  # Import the expand further module
from module_essay_writing import (
    reset_essay_session,
    generate_brainstorming_qna,
    generate_essay_structure,
    get_essay_feedback,
)

# Set page configuration with wide layout
st.set_page_config(page_title="வினவி", page_icon="📝", layout='wide')

# Load and encode the background image
image_path = "data/img.jpg"  # Update with your image path
if os.path.exists(image_path):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()
else:
    encoded_image = ""

# Add custom CSS for styling
st.markdown(f"""
    <style>
        body {{
            background-image: url('data:image/jpg;base64,{encoded_image}');
            background-size: cover;
            background-attachment: fixed;
        }}
        .main-content {{
            background-color: rgba(255, 255, 255, 0.8);
            padding: 2rem;
            border-radius: 10px;
        }}
        .chat-history {{
            max-height: 400px;
            overflow-y: auto;
            padding: 10px;
            margin-bottom: 20px;
        }}
        .chat-message {{
            margin-bottom: 10px;
        }}
        .user-message, .assistant-message {{
            padding: 10px;
            border-radius: 10px;
        }}
        .separator {{
            text-align: center;
            color: #888;
            margin: 20px 0;
        }}
        .stButton>button {{
            width: 100%;
            height: 40px;
            font-size: 16px;
            border-radius: 8px;
        }}
        .submit-button {{
            width: 100%;
            height: 40px;
            font-size: 16px;
            border-radius: 8px;
            margin-left: 10px;
        }}
        .read-aloud-button {{
            width: 100% !important;
        }}
        .input-row {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 10px;
        }}
        .input-col {{
            flex: 1;
        }}
        .mic-col {{
            margin-left: 10px;
        }}
        .sidebar-chat {{
            max-height: 300px;
            overflow-y: auto;
        }}
        .sidebar-line {{
            border-top: 1px solid #ddd;
            margin: 10px 0;
        }}
    </style>
""", unsafe_allow_html=True)

# Center Title and Description
st.markdown(
    "<h1 style='text-align: center;'>வினவி</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center;'> "
    "தமிழ் தொடர்புடைய எந்த கேள்வியையும் கேளுங்கள்.</p>",
    unsafe_allow_html=True
)

api_key = os.getenv("OPENAI_API_KEY")

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

if 'selected_option' not in st.session_state:
    st.session_state['selected_option'] = None

if 'last_answer' not in st.session_state:
    st.session_state['last_answer'] = ''

if 'is_processing' not in st.session_state:
    st.session_state['is_processing'] = False

if 'is_melum_kooru_active' not in st.session_state:
    st.session_state['is_melum_kooru_active'] = False

if 'melum_kooru_messages' not in st.session_state:
    st.session_state['melum_kooru_messages'] = []

if 'input_placeholder' not in st.session_state:
    st.session_state['input_placeholder'] = ''

if 'mic_counter' not in st.session_state:
    st.session_state['mic_counter'] = 0

# New session state variables for the new modes
if 'karutharithal_exercise' not in st.session_state:
    st.session_state['karutharithal_exercise'] = None  # To store the passage and questions

if 'nirappugaa_exercise' not in st.session_state:
    st.session_state['nirappugaa_exercise'] = None  # To store the passage with blanks

if 'karutharithal_started' not in st.session_state:
    st.session_state['karutharithal_started'] = False

if 'nirappugaa_started' not in st.session_state:
    st.session_state['nirappugaa_started'] = False

if 'user_answers' not in st.session_state:
    st.session_state['user_answers'] = []  # To store the user's answers

if 'exercise_feedback' not in st.session_state:
    st.session_state['exercise_feedback'] = ''  # To store feedback after validation

# Essay Writing Mode Variables
if 'essay_step' not in st.session_state:
    st.session_state['essay_step'] = 0  # To track the current step

if 'essay_title' not in st.session_state:
    st.session_state['essay_title'] = ''

if 'brainstorming_qna' not in st.session_state:
    st.session_state['brainstorming_qna'] = ''

if 'essay_structure' not in st.session_state:
    st.session_state['essay_structure'] = ''

if 'essay_content' not in st.session_state:
    st.session_state['essay_content'] = ''

if 'essay_feedback' not in st.session_state:
    st.session_state['essay_feedback'] = ''

if 'essay_mode_started' not in st.session_state:
    st.session_state['essay_mode_started'] = False



# --- Add this block to handle mode change ---
# Initialize previous mode
if 'prev_mode' not in st.session_state:
    st.session_state['prev_mode'] = None

# Sidebar: Mode Selection and Overall Chat History
with st.sidebar:
    st.write("## முறை தேர்ந்தெடுக்கவும் (Select Mode)")
    mode = st.radio(
        "",
        ("தமிழ் பயிற்சி", "நிரப்புக பயிற்சி", "விரிவாக"),
        disabled=st.session_state['is_processing']
    )

    # Separator line
    st.markdown("<div class='sidebar-line'></div>", unsafe_allow_html=True)

    # Overall Chat History with Scrollbar
    st.write("## உரையாடல் வரலாறு (Chat History)")
    chat_history_container = st.container()
    with chat_history_container:
        st.markdown("<div class='sidebar-chat'>", unsafe_allow_html=True)
        for message in st.session_state['messages']:
            role = "நீங்கள்" if message['role'] == 'user' else "உதவியாளர்"
            st.write(f"**{role}:** {message['content']}")
            st.markdown("<div class='separator'></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# Handle mode change
if st.session_state['prev_mode'] != mode:
    # Mode has changed
    # Reset variables associated with previous mode

    # Reset common variables
    st.session_state['selected_option'] = None
    st.session_state['last_answer'] = ''
    st.session_state['input_placeholder'] = ''
    st.session_state['is_processing'] = False

    # Reset variables specific to modes

    # Virivaaga mode variables
    st.session_state['is_melum_kooru_active'] = False
    st.session_state['melum_kooru_messages'] = []

    # Karutharithal mode variables
    st.session_state['karutharithal_exercise'] = None
    st.session_state['karutharithal_started'] = False

    # Nirappug mode variables
    st.session_state['nirappugaa_exercise'] = None
    st.session_state['nirappugaa_started'] = False

    # User answers and exercise feedback
    st.session_state['user_answers'] = []
    st.session_state['exercise_feedback'] = ''

    # Update the previous mode
    st.session_state['prev_mode'] = mode

# Function for content moderation
def moderate_content(user_input: str) -> bool:
    moderation_prompt = """
You are an assistant that checks if a user's input is appropriate for a 9-year-old child in Singapore in Tamil and English languages.
You have to be very accurate in flagging tamil/English bad words, inappropriate words and politically wrong words/phrases.
Your task is to analyze the input and determine if it contains any inappropriate, abusive, or exploitative content.
If the input is inappropriate for a child, respond with "Yes". If the input is appropriate, 
respond with "No".

User Input: {user_input}

Is the user input inappropriate for a 9-year-old child? (Yes/No):
"""
    moderation_llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0.0,
        max_tokens=5,
        openai_api_key=api_key
    )
    prompt = PromptTemplate(
        input_variables=["user_input"],
        template=moderation_prompt
    )
    formatted_prompt = prompt.format(user_input=user_input)
    response = moderation_llm.predict(formatted_prompt).strip().lower()
    return response.startswith('yes')

def autoplay_audio(text):
    # Remove '**' used for bold text
    cleaned_text = text.replace('**', '')
    cleaned_text = cleaned_text.replace('__', '')
    cleaned_text = cleaned_text.replace('<br>', '')



    # Convert the cleaned text to speech
    tts = gTTS(cleaned_text, lang='ta')
    audio_fp = io.BytesIO()
    tts.write_to_fp(audio_fp)
    audio_fp.seek(0)
    audio_bytes = audio_fp.read()
    b64 = base64.b64encode(audio_bytes).decode()
    
    # Embed the audio element in HTML for playback
    html_string = f"""
        <audio controls autoplay>
        <source src="data:audio/mpeg;base64,{b64}" type="audio/mpeg">
        </audio>
    """
    components.html(html_string, height=60)

# Helper function to format text with bullets
def format_with_bullets(text):
    lines = text.strip().split('\n')
    formatted_lines = [f"• {line.strip()}" for line in lines if line.strip()]
    return '<br>'.join(formatted_lines)

if mode == "நிரப்புக பயிற்சி":
    # Function to reset Nirappug session state
    def reset_nirappug_session():
        st.session_state['nirappugaa_started'] = False
        st.session_state['nirappugaa_exercise'] = None
        st.session_state['melum_kooru_messages'] = []
        st.session_state['messages'] = []
        st.session_state['exercise_feedback'] = ''
        st.session_state['user_answers'] = []
        st.session_state['is_processing'] = False

    # Callback function for starting Nirappug exercise
    def start_nirappugaa():
        try:
            with st.spinner("பயிற்சி தயாராகிறது..."):
                exercise = generate_nirappugaa_exercise(api_key)
                # Check if exercise has non-empty passage, blanks, and options
                if not exercise.get('passage') or not exercise.get('blanks') or not exercise.get('options'):
                    raise ValueError("பகுதி, குறைவுகள், அல்லது விருப்பங்கள் காலியாக உள்ளன. தயவுசெய்து மீண்டும் முயற்சிக்கவும்.")
                st.session_state['nirappugaa_exercise'] = exercise
                st.session_state['nirappugaa_started'] = True
        except ValueError as e:
            #st.error(str(e))
            reset_nirappug_session()
        except Exception as e:
            #st.error(f"பயிற்சி தயாரிப்பதில் ஒரு பிழை ஏற்பட்டது: {str(e)}")
            reset_nirappug_session()

    if not st.session_state.get('nirappugaa_started'):
        st.markdown(
            "<p style='text-align: center;'>நிரப்புக பயிற்சியை தொடங்குவோம். பயிற்சியை தொடங்க 'தொடங்கு' பொத்தானை அழுத்தவும்.</p>",
            unsafe_allow_html=True
        )
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.button("தொடங்கு", key='nirappugaa_start_btn', on_click=start_nirappugaa)
    else:
        # Check if the session state has the required data to avoid KeyError
        if 'nirappugaa_exercise' in st.session_state:
            exercise = st.session_state['nirappugaa_exercise']
            passage = exercise.get('passage', '')
            blanks = exercise.get('blanks', [])
            options = exercise.get('options', [])

            if not passage or not blanks or not options:
                #st.error("பயிற்சி தரவுகள் சரியாக இல்லை. மீண்டும் முயற்சிக்கவும்.")
                reset_nirappug_session()
            else:
                st.write("### பகுதி:")
                st.write(passage)

                # Add a "வாசிக்க" (Read Aloud) button below the "படிப்பு" section
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("வாசிக்க", key='nirapuga_read_aloud_btn'):
                        autoplay_audio(passage)

                # Input fields for blanks with options
                st.write("### குறைவுகள் நிரப்பவும்:")
                user_answers = []
                for idx in range(len(blanks)):
                    user_answer = st.selectbox(
                        f"பகுதி {idx + 1} - சரியான விடையை தேர்வு செய்யவும்:",
                        options=["------பதில் தேர்ந்தெடுக்கவும்------"] + options,
                        key=f'nirappugaa_answer_{idx}'
                    )
                    user_answers.append(user_answer)

                # Button to submit answers
                if st.button("பதில்கள் அனுப்பவும்", key='nirappugaa_submit_btn'):
                    # Validate the answers
                    st.session_state['is_processing'] = True
                    # Collect the answers
                    st.session_state['user_answers'] = user_answers
                    # Pass the answers through content moderation
                    inappropriate = False
                    for answer in user_answers:
                        if moderate_content(answer) and answer != "பதில் தேர்ந்தெடுக்கவும்":
                            inappropriate = True
                            break
                    if inappropriate:
                        st.error("உங்கள் பதில்களில் தவறான அல்லது பொருத்தமற்ற உள்ளடக்கம் உள்ளது. தயவுசெய்து சரிசெய்து மீண்டும் முயற்சிக்கவும்.")
                        st.session_state['is_processing'] = False
                    else:
                        with st.spinner("உங்கள் பதில்கள் மதிப்பாய்வு செய்யப்படுகின்றன..."):
                            # Validate the answers
                            feedback = validate_nirappugaa_answers(
                                passage,
                                blanks,
                                st.session_state['user_answers'],
                                api_key
                            )
                            st.session_state['exercise_feedback'] = feedback
                        st.session_state['is_processing'] = False

                # Display feedback
                if st.session_state['exercise_feedback']:
                    st.write("### மதிப்பாய்வு:")
                    st.write(st.session_state['exercise_feedback'])
                    if st.button("புதிய பயிற்சி", key='nirappugaa_new_exercise_btn'):
                        reset_nirappug_session()
        else:
            st.error("பயிற்சி தரவுகள் காணப்படவில்லை. தயவுசெய்து மீண்டும் தொடங்கவும்.")
            reset_nirappug_session()


elif mode == "விரிவாக":
    # Virivaaga Mode Implementation
    st.session_state['selected_option'] = 'virivaaga'

    # Initialize session state variables if not already done
    if 'melum_kooru_messages' not in st.session_state:
        st.session_state['melum_kooru_messages'] = []
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []
    if 'is_melum_kooru_active' not in st.session_state:
        st.session_state['is_melum_kooru_active'] = False
    if 'main_answer' not in st.session_state:
        st.session_state['main_answer'] = ''
    if 'melum_kooru_answers' not in st.session_state:
        st.session_state['melum_kooru_answers'] = []

    # Create a horizontal layout for mic button and text input
    st.markdown('<div class="input-row">', unsafe_allow_html=True)
    input_col,mic_col = st.columns([5, 1])  # Mic column first, then input

    with mic_col:
        tamil_text = speech_to_text(
            language='ta-IN',
            start_prompt="🎤",
            stop_prompt="🛑",
            key='STT_virivaaga'
        )
        if tamil_text:
            st.session_state['input_placeholder'] = tamil_text  # Update session state

    with input_col:
        user_input = st.text_input(
            "உங்கள் கேள்வியை இங்கே தட்டச்சு செய்யவும்:", 
            value=st.session_state.get('input_placeholder', ''), 
            key='virivaaga_user_input',
            disabled=st.session_state['is_processing']
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # Submit button centered below input
    st.write("")
    submit_col = st.columns([1, 2, 1])
    with submit_col[1]:
        submit_button = st.button('அனுப்பவும்', key='submit_btn', disabled=st.session_state['is_processing'])

    # Processing logic
    if submit_button and user_input and st.session_state.get('selected_option'):
        st.session_state['is_processing'] = True
        st.session_state['messages'].append({"role": "user", "content": user_input})
        selected_option = st.session_state['selected_option']

        # Check for inappropriate content
        with st.spinner("சரிபார்க்கிறது..."):
            if moderate_content(user_input):
                predefined_response = "மன்னிக்கவும், நான் அந்த கேள்விக்கு பதில் அளிக்க முடியாது."
                st.session_state['messages'].append({"role": "assistant", "content": predefined_response})
                st.session_state['main_answer'] = predefined_response
                st.session_state['melum_kooru_answers'] = []  # Clear previous melum kooru answers
                st.error(predefined_response)
            else:
                with st.spinner("சிந்திக்கிறது..."):
                    if selected_option == 'virivaaga':
                        conversation_chain = setup_melum_kooru_chain()
                        if st.session_state.get('is_melum_kooru_active'):
                            # Use the conversation history in 'melum_kooru_messages'
                            st.session_state['melum_kooru_messages'].append({"role": "user", "content": user_input})
                            conversation_history = '\n'.join([
                                f"{msg['role']}: {msg['content']}" for msg in st.session_state['melum_kooru_messages']
                            ])
                            answer = conversation_chain.run(input=conversation_history)
                            st.session_state['melum_kooru_messages'].append({"role": "assistant", "content": answer})
                        else:
                            # Start a new conversation
                            answer = conversation_chain.run(input=user_input)
                            st.session_state['is_melum_kooru_active'] = True
                            st.session_state['melum_kooru_messages'] = [
                                {"role": "user", "content": user_input},
                                {"role": "assistant", "content": answer}
                            ]
                    
                        # Format the assistant's answer with bullets
                        formatted_answer = format_with_bullets(answer)
                        # Append the assistant's answer to messages
                        st.session_state['messages'].append({"role": "assistant", "content": formatted_answer})
                        st.session_state['main_answer'] = formatted_answer  # Set main answer
                        st.session_state['melum_kooru_answers'] = []  # Clear any previous melum kooru answers
                    else:
                        st.error("தவறான விருப்பம் தேர்ந்தெடுக்கப்பட்டது.")
                        #st.stop()

        st.session_state['input_placeholder'] = ''  # Reset input after processing
        st.session_state['is_processing'] = False

    # Display assistant's answer and action buttons
    if st.session_state['main_answer']:
        st.write("## உதவியாளர் பதில்")
        st.markdown(f"<div class='chat-message assistant-message'>{st.session_state['main_answer']}</div>", unsafe_allow_html=True)
        st.markdown("<p>மேலும் அறிய 'மேலும் கூரு' பொத்தானைக் கிளிக் செய்யவும் அல்லது உங்கள் கேள்வியை உள்ளீடு செய்து அனுப்பவும்.</p>", unsafe_allow_html=True)

        if mode == "விரிவாக":
            # Create a container for the buttons to keep them aligned
            with st.container():
                # Create two columns for the action buttons
                button_col1, button_col2 = st.columns(2)
                with button_col1:
                    if st.button("வாசிக்க", key='virivaaga_read_aloud_btn', disabled=st.session_state['is_processing']):
                        autoplay_audio(st.session_state['main_answer'])
                with button_col2:
                    if st.button("மேலும் கூரு", key='virivaaga_melum_kooru_btn', disabled=st.session_state['is_processing']):
                        # Call the expand further pipeline
                        expand_chain = setup_expand_further_chain()  # Initialize the chain

                        # Prepare the conversation history and last assistant message for expansion
                        last_assistant_message = st.session_state['main_answer']
                        conversation_history = "\n".join([
                            f"{msg['role']}: {msg['content']}" for msg in st.session_state['melum_kooru_messages'][-10:]
                        ])

                        # Get the expanded response
                        with st.spinner("மேலும் சிந்திக்கிறது..."):
                            expanded_answer = expand_chain.run({
                                "conversation_history": conversation_history,
                                "last_assistant_message": last_assistant_message
                            })

                        # Format the expanded answer with bullets
                        formatted_expanded_answer = format_with_bullets(expanded_answer)

                        # Add the user's "Yes" response and the assistant's expanded answer to the conversation
                        st.session_state['melum_kooru_messages'].append({"role": "user", "content": "ஆம்"})
                        st.session_state['melum_kooru_messages'].append({"role": "assistant", "content": formatted_expanded_answer})
                        st.session_state['messages'].append({"role": "assistant", "content": formatted_expanded_answer})
                        st.session_state['melum_kooru_answers'].append(formatted_expanded_answer)
                        # No need to set 'last_answer' here

            # Display the expanded answers below the main answer
            for expanded_answer in st.session_state['melum_kooru_answers']:
                st.markdown(f"<div class='chat-message assistant-message'>{expanded_answer}</div>", unsafe_allow_html=True)

    else:
        st.write("")

elif mode == "தமிழ் பயிற்சி":
    # Handle the "Tamil Udhavi (Tamil Assistance)" mode

    # Display input bar first
    st.markdown('<div class="input-row">', unsafe_allow_html=True)
    input_col, mic_col = st.columns([5, 1])
    with mic_col:
        tamil_text = speech_to_text(
            language='ta-IN',
            start_prompt="🎤",
            stop_prompt="🛑",
            key='STT_tamil_udhavi'
        )
        if tamil_text:
            st.session_state['input_placeholder'] = tamil_text  # Update session state

    with input_col:
        user_input = st.text_input(
            "உங்கள் கேள்வியை இங்கே தட்டச்சு செய்யவும்:", 
            value=st.session_state.get('input_placeholder', ''), 
            key='tamil_udhavi_user_input',
            disabled=st.session_state['is_processing']
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # Display options buttons below the input bar
    st.markdown("<p style='font-size:16px;'>தேர்வு செய்ய ஒரு விருப்பம்</p>", unsafe_allow_html=True)

    button_labels = ["பொருள்", "உதாரணம்", "மொழிபெயர்ப்பு"]
    button_keys = ['meaning', 'example', 'translation']

    button_cols = st.columns(len(button_labels))
    for idx, (col, label, key) in enumerate(zip(button_cols, button_labels, button_keys)):
        with col:
            button_clicked = st.button(label, key=f'tamil_udhavi_{key}', disabled=st.session_state['is_processing'])
            if button_clicked and user_input:
                st.session_state['is_processing'] = True
                # Append to messages for user input
                st.session_state['messages'].append({"role": "user", "content": user_input})
                st.session_state['selected_option'] = key

                # Check for inappropriate content
                with st.spinner("சரிபார்க்கிறது..."):
                    if moderate_content(user_input):
                        predefined_response = "மன்னிக்கவும், நான் அந்த கேள்விக்கு பதில் அளிக்க முடியாது."
                        st.session_state['messages'].append({"role": "assistant", "content": predefined_response})
                        st.session_state['last_answer'] = predefined_response
                        st.error(predefined_response)
                    else:
                        with st.spinner("சிந்திக்கிறது..."):
                            if key == 'meaning':
                                qa_chain = setup_rag_pipeline_meaning()
                                result = qa_chain({"query": user_input})
                                answer = result['result']
                            elif key == 'example':
                                qa_chain = setup_rag_pipeline_example()
                                result = qa_chain({"query": user_input})
                                answer = result['result']
                            elif key == 'translation':
                                translation_chain = setup_translation_chain()
                                answer = translation_chain.run(question=user_input)
                            else:
                                st.error("தவறான விருப்பம் தேர்ந்தெடுக்கப்பட்டது.")
                                #st.stop()

                            # Format the assistant's answer with bullets
                            formatted_answer = format_with_bullets(answer)

                            # Append the assistant's answer to messages
                            st.session_state['messages'].append({"role": "assistant", "content": formatted_answer})
                            st.session_state['last_answer'] = formatted_answer

                st.session_state['input_placeholder'] = ''  # Reset input after processing
                st.session_state['is_processing'] = False

    # Display assistant's answer
    if st.session_state['last_answer']:
        st.write("## உதவியாளர் பதில்")
        st.markdown(f"<div class='chat-message assistant-message'>{st.session_state['last_answer']}</div>", unsafe_allow_html=True)

        # For Tamil Udhavi mode, only display the "வாசிக்க" button
        if st.button("வாசிக்க", key='tamil_udhavi_read_aloud_btn', disabled=st.session_state['is_processing']):
            autoplay_audio(st.session_state['last_answer'])
    else:
        st.write("")

# Note: No additional code needed for other modes as they are already handled above.
