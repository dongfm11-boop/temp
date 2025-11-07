import streamlit as st
from google import genai
from google.genai import types
import pandas as pd
import io
from datetime import datetime
import time

# --- Configuration and Initialization ---
APP_TITLE = "👕 Streamlit 기반 의상 추천 AI 챗봇"
DEFAULT_MODEL = "gemini-2.0-flash"
SYSTEM_PROMPT = """
당신은 고객의 상황에 맞는 옷차림을 간결하게 추천해주는 친절한 AI 스타일리스트입니다.
고객은 외출 전 바쁜 상황이므로, 대답은 반드시 간결하고 실용적이어야 합니다.

[운영 규칙]
1. 사용자가 오늘의 온도, 일교차, 어떤 활동을 하는지, 성별, 나이 등의 정보를 제공하면, 이 정보를 **구체적으로** 정리하여 수집하세요.
2. 정보가 충분하지 않으면, 부족한 부분을 한 문장으로 **간결하게** 되물어보세요.
3. 필요한 정보를 모두 수집한 후에는, 사용자의 상황(온도/활동/성별/나이 등)에 맞는 **실용적인 옷차림**을 한 문단으로 **간결하게** 추천하세요.
4. 마지막에는 반드시 다음 메시지를 출력하세요: "오늘의 당신에게 맞는 상품을 추천해드릴 수 있어요."라고 정중히 안내하세요.
"""
RESTART_MESSAGE = "❗️ API 요청 오류(429 등)가 발생하여 이전 6턴의 대화만 유지하고 채팅 세션을 재시작합니다."
MAX_HISTORY_TURN = 6 # Max turns (user/model pair) to keep upon 429 restart

# Available models list (excluding -exp)
AVAILABLE_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gemini-2.0-pro",
    "gemini-2.5-pro",
]

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

# --- State Initialization Helpers ---

def get_api_key():
    """Load API key from st.secrets or prompt user input."""
    api_key = st.secrets.get('GEMINI_API_KEY')
    if not api_key:
        api_key = st.sidebar.text_input("Gemini API Key", type="password", help="Streamlit Secrets에 설정되지 않은 경우 여기에 입력하세요.")
        
    if not api_key:
        st.info("왼쪽 사이드바에 Gemini API 키를 입력하거나 Streamlit Secrets에 설정해야 합니다.")
        st.stop()
        
    return api_key

def initialize_client_and_chat(api_key, model_name, system_prompt, history_to_restore=None):
    """Initializes Gemini client and a new Chat session."""
    try:
        client = genai.Client(api_key=api_key)
        
        # System instruction configuration
        config = types.GenerateContentConfig(
            system_instruction=system_prompt
        )
        
        # Start new Chat session
        chat = client.chats.create(model=model_name, config=config)
        st.session_state.gemini_chat = chat
        st.session_state.model_name = model_name
        
        # Restore history if provided (used for 429 restart)
        if history_to_restore:
            # Reconstruct Chat history for the new session
            for msg in history_to_restore:
                # Map Streamlit role to Gemini role
                role_map = {"user": "user", "assistant": "model"}
                chat.history.append(
                    types.Content(
                        role=role_map[msg["role"]],
                        parts=[types.Part.from_text(msg["content"])]
                    )
                )
            st.session_state.messages = history_to_restore
            st.session_state.messages.append({"role": "assistant", "content": RESTART_MESSAGE})
            st.session_state.history_log.append({
                "Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "Role": "assistant",
                "Content": RESTART_MESSAGE,
                "Model": st.session_state.model_name
            })
            
        else:
            st.session_state.messages = []
            st.session_state.history_log = [] # Full conversation log for CSV
            
        st.rerun()
        
    except Exception as e:
        st.error(f"Gemini 클라이언트/채팅 초기화 오류: {e}")
        st.stop()


# 3. Initial session setup if not exists
if 'gemini_chat' not in st.session_state:
    st.session_state.messages = []
    st.session_state.history_log = []
    st.session_state.model_name = DEFAULT_MODEL
    
# --- Sidebar and UI Configuration ---

with st.sidebar:
    st.header("설정 및 도구")
    
    # Model Selection
    selected_model = st.selectbox(
        "사용할 모델 선택", 
        AVAILABLE_MODELS,
        index=AVAILABLE_MODELS.index(DEFAULT_MODEL),
        key="model_select_key"
    )
    
    # API Key Load (Stops if not available)
    api_key = get_api_key()
    
    # Session Reset Button
    if st.button("💬 대화 초기화 및 모델 적용", help="대화 기록을 지우고 새 모델로 시작합니다."):
        initialize_client_and_chat(api_key, selected_model, SYSTEM_PROMPT)

    st.markdown("---")
    st.subheader("로그 기록 옵션")
    
    # CSV Logging Option
    if 'auto_log' not in st.session_state:
        st.session_state.auto_log = False
    st.session_state.auto_log = st.checkbox("CSV 자동 기록 (대화마다 기록)", st.session_state.auto_log)
    
    # Log Download
    if st.session_state.history_log:
        log_df = pd.DataFrame(st.session_state.history_log)
        # Convert to CSV and ensure proper encoding
        csv_data = log_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ 대화 로그 CSV 다운로드",
            data=csv_data,
            file_name=f"gemini_chat_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
    else:
        st.info("로그 기록이 없습니다.")

# Client and Chat session check (Re-initialize if model changed or first run)
if 'gemini_chat' not in st.session_state or st.session_state.model_name != selected_model:
    if api_key:
        initialize_client_and_chat(api_key, selected_model, SYSTEM_PROMPT)

# Display Model and Session Info
st.sidebar.markdown("---")
st.sidebar.markdown(f"**현재 모델:** `{st.session_state.model_name}`")
st.sidebar.markdown(f"**총 턴 수:** {len(st.session_state.messages)//2} (메시지 {len(st.session_state.messages)}개)")


# --- Display Conversation History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- User Input Processing ---
if prompt := st.chat_input("오늘의 날씨와 일정을 말씀해 주세요 (예: 오늘 최고 25도, 최저 10도, 친구와 카페에 갑니다, 여성, 30대)"):
    # 1. Record and Display User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Log User Message
    st.session_state.history_log.append({
        "Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "Role": "user",
        "Content": prompt,
        "Model": st.session_state.model_name
    })
    
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Call Gemini API
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # Use send_message for continuous conversation
            response = st.session_state.gemini_chat.send_message(prompt, stream=True)
            
            # Stream the response
            for chunk in response:
                if chunk.text:
                    full_response += chunk.text
                    message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
        
        except types.errors.ResourceExhaustedError: # Handle 429
            # Get the last MAX_HISTORY_TURN pairs
            history_to_keep = st.session_state.messages[-(MAX_HISTORY_TURN * 2):]
            log_to_keep = st.session_state.history_log[-(MAX_HISTORY_TURN * 2):]

            # Re-initialize client/chat, restoring history
            initialize_client_and_chat(api_key, st.session_state.model_name, SYSTEM_PROMPT, history_to_keep)
            # Rerun will happen inside the helper function
            
        except Exception as e:
            full_response = f"죄송합니다. 오류가 발생했습니다: {e}"
            message_placeholder.markdown(full_response)
            
        # 3. Record Model Response
        if full_response and full_response != RESTART_MESSAGE:
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
            # Log Assistant Message (if not a restart message)
            if st.session_state.auto_log:
                st.session_state.history_log.append({
                    "Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "Role": "assistant",
                    "Content": full_response,
                    "Model": st.session_state.model_name
                })
        
    st.rerun() # Refresh to show updated chat