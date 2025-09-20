import urllib.parse
from dotenv import load_dotenv
import os
import json
import asyncio
import logging
import threading
import queue
import argparse
import io
from typing import List, Dict, Any, Optional, Generator, Tuple
import numpy as np
import wave
import struct
from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_tool_calling_agent, AgentExecutor
from fastrtc import (
    AlgoOptions,
    ReplyOnPause,
    Stream,
    audio_to_bytes,
)
from elevenlabs.client import ElevenLabs
from loguru import logger as voice_logger

# Add pydub for MP3 handling - install with: pip install pydub
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    print("WARNING: pydub not installed. Install with: pip install pydub")
    print("MP3 audio will be converted to silence as fallback")

# Remove default loguru handler and add custom format
voice_logger.remove()
voice_logger.add(
    lambda msg: print(msg),
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <level>{message}</level>",
)

# Available Coral tools
LIST_AGENTS_TOOL = "list_agents"
CREATE_THREAD_TOOL = "create_thread"
SEND_MESSAGE_TOOL = "send_message"
WAIT_FOR_MENTIONS_TOOL = "wait_for_mentions"
ADD_PARTICIPANT_TOOL = "add_participant"
CLOSE_THREAD_TOOL = "close_thread"

MAX_CHAT_HISTORY = 10
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 8000
SLEEP_INTERVAL = 1
ERROR_RETRY_INTERVAL = 5
WAIT_TIMEOUT_MS = 30000

# Setup standard logging for the main system
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Global variables for sharing between voice handler and main system
elevenlabs_client = None  # For both STT and TTS
agent_executor = None
agent_tools = {}
chat_history = []
current_thread_id = None

class UserInputHandler:
    def __init__(self):
        self.input_queue = queue.Queue()
        self.running = True
    
    def start_input_thread(self):
        """Start a separate thread to handle user input"""
        def input_worker():
            print("\n" + "="*60)
            print("CRISIS SUPPORT TRIAGE AGENT - USER INPUT INTERFACE")
            print("="*60)
            print("You can now simulate caller interactions with the triage agent.")
            print("Type messages as if you are someone calling the crisis support line.")
            print("The agent will assess your situation and connect you with counselors.")
            print("Type 'quit' or 'exit' to stop the program.")
            print("="*60 + "\n")
            
            while self.running:
                try:
                    user_input = input("Caller: ").strip()
                    if user_input.lower() in ['quit', 'exit', 'stop']:
                        print("Stopping the triage agent...")
                        self.running = False
                        self.input_queue.put(None)
                        break
                    elif user_input:
                        self.input_queue.put(user_input)
                        print(f"Message received: '{user_input}'")
                    else:
                        print("Please enter a message or 'quit' to exit.")
                except EOFError:
                    print("\nInput ended, stopping agent...")
                    self.running = False
                    self.input_queue.put(None)
                    break
                except KeyboardInterrupt:
                    print("\nInterrupted by user, stopping agent...")
                    self.running = False
                    self.input_queue.put(None)
                    break
        
        thread = threading.Thread(target=input_worker, daemon=True)
        thread.start()
        return thread
    
    def get_input(self):
        """Get user input from queue (non-blocking)"""
        try:
            return self.input_queue.get_nowait()
        except queue.Empty:
            return None
    
    def stop(self):
        """Stop the input handler"""
        self.running = False

def load_config() -> Dict[str, Any]:
    runtime = os.getenv("CORAL_ORCHESTRATION_RUNTIME", None)
    if runtime is None:
        load_dotenv()
    
    config = {
        "runtime": os.getenv("CORAL_ORCHESTRATION_RUNTIME", None),
        "coral_sse_url": os.getenv("CORAL_SSE_URL"),
        "agent_id": os.getenv("CORAL_AGENT_ID", "triage_interface_agent"),
        "model_name": os.getenv("MODEL_NAME"),
        "model_provider": os.getenv("MODEL_PROVIDER"),
        "groq_api_key": os.getenv("OPENAI_API_KEY"),  # Keep for model, not audio
        "elevenlabs_api_key": os.getenv("ELEVENLABS_API_KEY"),
        "model_temperature": float(os.getenv("MODEL_TEMPERATURE", DEFAULT_TEMPERATURE)),
        "model_token": int(os.getenv("MODEL_MAX_TOKENS", DEFAULT_MAX_TOKENS)),
        "base_url": os.getenv("BASE_URL"),
        "elevenlabs_voice_id": os.getenv("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb"),  # Default to George voice
        "elevenlabs_model": os.getenv("ELEVENLABS_MODEL", "eleven_multilingual_v2")
    }
    
    required_fields = ["coral_sse_url", "agent_id", "model_name", "model_provider", "groq_api_key", "elevenlabs_api_key"]
    missing = [field for field in required_fields if not config[field]]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
    
    if not 0 <= config["model_temperature"] <= 2:
        raise ValueError(f"Model temperature must be between 0 and 2, got {config['model_temperature']}")
    
    if config["model_token"] <= 0:
        raise ValueError(f"Model token must be positive, got {config['model_token']}")
    
    return config

def get_tools_description(tools: List[Any]) -> str:
    descriptions = []
    for tool in tools:
        tool_desc = f"Tool: {tool.name}, Schema: {json.dumps(tool.args).replace('{', '{{').replace('}', '}}')}"
        descriptions.append(tool_desc)
    
    return "\n".join(descriptions)

def format_chat_history(chat_history: List[Dict[str, str]]) -> str:
    if not chat_history:
        return "No previous chat history available."
    
    history_str = "Previous Conversations (use this to understand caller context and risk assessment):\n"
    
    for i, chat in enumerate(chat_history, 1):
        history_str += f"Conversation {i}:\n"
        history_str += f"Caller: {chat['user_input']}\n"
        history_str += f"Triage Agent: {chat['response']}\n\n"
    
    return history_str

async def find_available_counselors(agent_tools: Dict[str, Any]) -> List[str]:
    """Find available counselor agents"""
    try:
        result = await agent_tools[LIST_AGENTS_TOOL].ainvoke({
            "includeDetails": True
        })
        
        counselor_ids = []
        if isinstance(result, dict):
            agents = result.get('agents', [])
            for agent in agents:
                agent_id = agent.get('id', '')
                description = agent.get('description', '').lower()
                if any(keyword in description for keyword in ['counselor', 'therapist', 'crisis', 'support', 'discord']):
                    counselor_ids.append(agent_id)
        
        logger.info(f"Found {len(counselor_ids)} potential counselors: {counselor_ids}")
        return counselor_ids
        
    except Exception as e:
        logger.error(f"Error finding counselors: {str(e)}")
        return []

async def create_counselor_thread(agent_tools: Dict[str, Any], caller_context: str) -> Optional[str]:
    """Create a thread with available counselors"""
    try:
        counselor_ids = await find_available_counselors(agent_tools)
        
        if not counselor_ids:
            logger.warning("No counselors available")
            return None
        
        result = await agent_tools[CREATE_THREAD_TOOL].ainvoke({
            "threadName": f"Crisis Support - {asyncio.get_event_loop().time()}",
            "participantIds": counselor_ids
        })
        
        if isinstance(result, dict):
            thread_id = result.get('threadId')
            if thread_id:
                context_message = f"HANDOFF FROM TRIAGE: {caller_context}"
                await agent_tools[SEND_MESSAGE_TOOL].ainvoke({
                    "threadId": thread_id,
                    "content": context_message,
                    "mentions": counselor_ids
                })
                logger.info(f"Created counselor thread: {thread_id}")
                return thread_id
        
        return None
        
    except Exception as e:
        logger.error(f"Error creating counselor thread: {str(e)}")
        return None

async def process_triage_input(user_input: str) -> str:
    """Process input through the triage agent and return response"""
    global agent_executor, agent_tools, chat_history, current_thread_id
    
    try:
        formatted_history = format_chat_history(chat_history)
        
        result = await agent_executor.ainvoke({
            "user_input": user_input,
            "agent_scratchpad": [],
            "chat_history": formatted_history
        })
        
        response = result.get('output', 'I apologize, but I need to connect you with a counselor right away. Please hold on.')
        
        # Update chat history
        chat_history.append({"user_input": user_input, "response": response})
        if len(chat_history) > MAX_CHAT_HISTORY:
            chat_history.pop(0)
        
        # Check if we should create a counselor thread
        if len(chat_history) >= 2 and not current_thread_id:
            context_summary = f"Caller assessment: {chat_history[-1]['user_input'][:200]}..."
            current_thread_id = await create_counselor_thread(agent_tools, context_summary)
            
            if current_thread_id:
                handoff_message = " I'm now connecting you with a trained counselor who can provide you with the support you need."
                response += handoff_message
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing triage input: {str(e)}")
        return "I apologize for the technical difficulty. Let me connect you with a counselor immediately."

def audio_to_wav_bytes(audio_tuple: Tuple[int, np.ndarray]) -> bytes:
    """Convert audio tuple to WAV format bytes for ElevenLabs STT"""
    sample_rate, audio_data = audio_tuple
    
    # Ensure audio is in the right format
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)
    
    # Convert float32 [-1, 1] to int16 [-32768, 32767]
    audio_int16 = (audio_data * 32767).astype(np.int16)
    
    # Create WAV file in memory
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())
    
    wav_buffer.seek(0)
    return wav_buffer.read()

def decode_mp3_with_pydub(mp3_data: bytes) -> Tuple[int, np.ndarray]:
    """Decode MP3 data using pydub and convert to numpy array"""
    try:
        # Load MP3 data into AudioSegment
        audio_segment = AudioSegment.from_mp3(io.BytesIO(mp3_data))
        
        # Convert to mono if stereo
        if audio_segment.channels > 1:
            audio_segment = audio_segment.set_channels(1)
        
        # Get sample rate
        sample_rate = audio_segment.frame_rate
        
        # Convert to raw audio data
        raw_audio = audio_segment.raw_data
        
        # Convert to numpy array
        # AudioSegment uses 16-bit samples by default
        audio_array = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
        
        voice_logger.debug(f"MP3 decoded: {len(audio_array)} samples at {sample_rate} Hz")
        return sample_rate, audio_array
        
    except Exception as e:
        voice_logger.error(f"Error decoding MP3 with pydub: {str(e)}")
        # Return silence as fallback
        return 22050, np.zeros(22050 * 3, dtype=np.float32)  # 3 seconds of silence

def process_elevenlabs_tts(tts_response) -> Generator[Tuple[int, np.ndarray], None, None]:
    """Process ElevenLabs TTS response and convert to audio format expected by FastRTC"""
    try:
        voice_logger.debug("Processing ElevenLabs TTS response...")
        
        # ElevenLabs returns audio as bytes generator
        if hasattr(tts_response, '__iter__') and not isinstance(tts_response, (str, bytes)):
            # If it's a generator, collect all chunks
            audio_content = b''.join(tts_response)
        elif hasattr(tts_response, 'read'):
            # If it's a file-like object, read it
            audio_content = tts_response.read()
        elif isinstance(tts_response, bytes):
            audio_content = tts_response
        else:
            # Try to convert to bytes
            audio_content = bytes(tts_response)
        
        voice_logger.debug(f"Audio content length: {len(audio_content) if audio_content else 'None'}")
        
        if not audio_content:
            voice_logger.error("No audio content received from ElevenLabs TTS")
            # Return 2 seconds of silence
            for _ in range(int(24000 * 2 / 1024)):
                yield (24000, np.zeros(1024, dtype=np.float32))
            return
        
        # Detect audio format and process accordingly
        try:
            # Check if it's MP3 format (starts with ID3 or sync frame)
            if audio_content.startswith(b'ID3') or (len(audio_content) > 2 and audio_content[0:2] == b'\xff\xfb'):
                voice_logger.debug("Processing MP3 format from ElevenLabs")
                
                if PYDUB_AVAILABLE:
                    # Use pydub to decode MP3
                    sample_rate, audio_array = decode_mp3_with_pydub(audio_content)
                    voice_logger.debug("✅ MP3 successfully decoded with pydub")
                else:
                    # Fallback: estimate duration and create silence
                    voice_logger.warning("MP3 format detected but pydub not available - generating silence")
                    sample_rate = 22050
                    # Rough estimation: 1KB per second of audio at low quality
                    estimated_duration = len(audio_content) / 1000.0
                    estimated_duration = max(2.0, min(estimated_duration, 30.0))  # Clamp between 2-30 seconds
                    audio_array = np.zeros(int(sample_rate * estimated_duration), dtype=np.float32)
                    voice_logger.debug(f"Generated {estimated_duration:.1f}s of silence as MP3 fallback")
                
            elif audio_content.startswith(b'RIFF'):
                # WAV format - parse header
                voice_logger.debug("Processing WAV format from ElevenLabs")
                wav_buffer = io.BytesIO(audio_content)
                with wave.open(wav_buffer, 'rb') as wav_file:
                    sample_rate = wav_file.getframerate()
                    n_channels = wav_file.getnchannels()
                    sample_width = wav_file.getsampwidth()
                    frames = wav_file.readframes(wav_file.getnframes())
                
                # Convert to float32
                if sample_width == 2:  # 16-bit
                    audio_array = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                elif sample_width == 4:  # 32-bit
                    audio_array = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
                else:
                    raise ValueError(f"Unsupported sample width: {sample_width}")
                
                # Handle stereo to mono conversion
                if n_channels == 2:
                    audio_array = audio_array.reshape(-1, 2).mean(axis=1)
                
                voice_logger.debug("✅ WAV format processed successfully")
                
            else:
                # Assume raw PCM or other format
                voice_logger.debug("Processing raw PCM format")
                sample_rate = 22050  # Common rate for free tier
                
                try:
                    # Try as 16-bit PCM first
                    audio_array = np.frombuffer(audio_content, dtype=np.int16).astype(np.float32) / 32768.0
                except ValueError:
                    try:
                        # Try as 32-bit float
                        audio_array = np.frombuffer(audio_content, dtype=np.float32)
                    except ValueError:
                        # Last resort - create silence
                        voice_logger.warning("Could not parse audio data, generating silence")
                        duration = 3.0  # 3 seconds
                        audio_array = np.zeros(int(sample_rate * duration), dtype=np.float32)
        
        except Exception as e:
            voice_logger.error(f"Error parsing audio: {str(e)}")
            # Generate fallback silence
            sample_rate = 22050  # Use lower sample rate for compatibility
            duration = 3.0
            audio_array = np.zeros(int(sample_rate * duration), dtype=np.float32)
        
        voice_logger.debug(f"Final audio array shape: {audio_array.shape}, sample rate: {sample_rate} Hz")
        
        # Stream audio in chunks
        chunk_size = 1024
        total_chunks = 0
        
        for i in range(0, len(audio_array), chunk_size):
            chunk = audio_array[i:i + chunk_size]
            if len(chunk) < chunk_size:
                # Pad the last chunk if necessary
                padded_chunk = np.zeros(chunk_size, dtype=np.float32)
                padded_chunk[:len(chunk)] = chunk
                chunk = padded_chunk
            
            yield (sample_rate, chunk)
            total_chunks += 1
        
        voice_logger.debug(f"Streamed {total_chunks} chunks at {sample_rate} Hz")
        
    except Exception as e:
        voice_logger.error(f"Error processing ElevenLabs TTS: {str(e)}")
        # Return silence in case of error
        for _ in range(int(22050 * 2 / 1024)):
            yield (22050, np.zeros(1024, dtype=np.float32))

def voice_response(audio: Tuple[int, np.ndarray]) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Process voice input, transcribe it, generate triage response, and deliver TTS audio.
    """
    global elevenlabs_client
    
    if elevenlabs_client is None:
        voice_logger.error("ElevenLabs client not initialized")
        yield (22050, np.zeros(1024, dtype=np.float32))
        return
    
    voice_logger.info("🎙️ Received voice input from caller")
    
    try:
        # Convert audio to WAV format for ElevenLabs STT
        voice_logger.debug("🔄 Converting audio to WAV format...")
        wav_bytes = audio_to_wav_bytes(audio)
        
        # Transcribe audio using ElevenLabs
        voice_logger.debug("🔄 Transcribing audio with ElevenLabs...")
        audio_data = io.BytesIO(wav_bytes)
        
        transcription = elevenlabs_client.speech_to_text.convert(
            file=audio_data,
            model_id="scribe_v1",
            tag_audio_events=True,
            language_code="eng",
            diarize=False  # Single speaker for crisis calls
        )
        
        # Extract transcript text
        if hasattr(transcription, 'text'):
            transcript = transcription.text
        elif isinstance(transcription, dict):
            transcript = transcription.get('text', '')
        else:
            transcript = str(transcription)
        
        voice_logger.info(f'👂 Transcribed: "{transcript}"')
        
        if not transcript or transcript.strip() == "":
            voice_logger.warning("Empty transcript received")
            # Generate a prompt for the caller to speak
            response_text = "I'm here to listen. Please tell me what's going on."
        else:
            # Process through triage agent (run in event loop)
            voice_logger.debug("🧠 Running triage assessment...")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                response_text = loop.run_until_complete(process_triage_input(transcript))
            finally:
                loop.close()
        
        voice_logger.info(f'💬 Triage Response: "{response_text}"')
        
        # Generate TTS response using ElevenLabs (Free Tier Compatible with MP3 handling)
        voice_logger.debug("🔊 Generating speech response with ElevenLabs (Free Tier + MP3 Support)...")
        try:
            # Get voice configuration from environment
            voice_id = os.getenv("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")  # George voice (calm, supportive)
            model_id = os.getenv("ELEVENLABS_MODEL", "eleven_multilingual_v2")
            
            # Use MP3 format for free tier compatibility
            voice_logger.debug("Using MP3 format for free tier compatibility with proper decoding")
            audio_generator = elevenlabs_client.text_to_speech.convert(
                text=response_text,
                voice_id=voice_id,
                model_id=model_id,
                output_format="mp3_22050_32",  # Low-quality MP3 for free tier
            )
            
            voice_logger.debug("✅ ElevenLabs TTS generation successful (Free Tier + MP3 Decoder)")
            yield from process_elevenlabs_tts(audio_generator)
            
        except Exception as e:
            voice_logger.error(f"ElevenLabs TTS error: {str(e)}")
            # Try with even more basic settings as fallback
            try:
                voice_logger.info("Trying most basic TTS settings for free tier...")
                audio_generator = elevenlabs_client.text_to_speech.convert(
                    text=response_text,
                    voice_id=voice_id,
                    model_id="eleven_multilingual_v2",
                    # Don't specify output_format to use default (should be free tier compatible)
                )
                yield from process_elevenlabs_tts(audio_generator)
                
            except Exception as e2:
                voice_logger.error(f"All TTS attempts failed: {str(e2)}")
                # Generate clean silence with appropriate duration
                sample_rate = 22050  # Lower sample rate
                duration = max(3.0, len(response_text) * 0.08)  # Rough duration based on text length
                samples = int(sample_rate * duration)
                
                # Stream clean silence in chunks
                chunk_size = 1024
                voice_logger.info(f"Streaming {duration:.1f} seconds of silence as audio placeholder")
                for i in range(0, samples, chunk_size):
                    chunk_samples = min(chunk_size, samples - i)
                    yield (sample_rate, np.zeros(chunk_samples, dtype=np.float32))
        
    except Exception as e:
        voice_logger.error(f"Error in voice response: {str(e)}")
        # Generate error message
        try:
            error_message = "I'm having trouble hearing you clearly. Please hold while I connect you with a counselor."
            voice_id = os.getenv("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")
            
            error_audio = elevenlabs_client.text_to_speech.convert(
                text=error_message,
                voice_id=voice_id,
                model_id="eleven_multilingual_v2",
                # Use default format for maximum compatibility
            )
            yield from process_elevenlabs_tts(error_audio)
        except:
            # If everything fails, return silence
            voice_logger.error("All methods failed, returning silence")
            for _ in range(int(22050 * 3 / 1024)):  # 3 seconds of silence at lower sample rate
                yield (22050, np.zeros(1024, dtype=np.float32))

def create_voice_stream() -> Stream:
    """Create and configure a Stream instance for voice interactions."""
    return Stream(
        modality="audio",
        mode="send-receive",
        handler=ReplyOnPause(
            voice_response,
            algo_options=AlgoOptions(
                speech_threshold=0.3,  # Lower threshold for crisis situations
            ),
        ),
    )

async def create_agent(coral_tools: List[Any]) -> AgentExecutor:
    coral_tools_description = get_tools_description(coral_tools)
    
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            f"""
            You are a suicide hotline triage assistant. Your primary role is to conduct a brief, sequential safety assessment and connect callers to trained counselors using available coral tools.

            CRITICAL: You must ask questions ONE AT A TIME and wait for each response before proceeding to the next question. Do not ask multiple questions in a single message.

            ## CONVERSATION FLOW - FOLLOW STRICTLY IN ORDER:

            ### STEP 1: Initial Contact
            - Greet warmly: "Hi, thank you for reaching out. I'm here to listen."
            - Show empathy: Acknowledge their courage for calling
            - Explain briefly: "I'd like to ask you a few quick questions so I can connect you with the right counselor."
            - WAIT for their response before proceeding

            ### STEP 2: Safety Assessment (Ask ONE question at a time)
            
            **Question 1 ONLY:** "Are you safe right now?"
            - WAIT for their answer
            - Acknowledge their response empathetically
            - Only after receiving their answer, proceed to Question 2

            **Question 2 ONLY:** "Do you have thoughts of hurting yourself?"
            - WAIT for their answer
            - Acknowledge with empathy ("Thank you for telling me" or "I hear you")
            - Only after receiving their answer, proceed to Question 3

            **Question 3 ONLY:** "Do you have a plan to act on those thoughts?"
            - WAIT for their answer
            - Acknowledge empathetically
            - Only after receiving their answer, proceed to handoff

            ### STEP 3: Risk Assessment & Tool Usage

            Based on their answers, determine risk level and use tools:

            **High Risk** (Yes to Q2 or Q3):
            - Say: "Thank you for telling me. This sounds very serious. I'm connecting you right now with a trained counselor."
            - IMMEDIATELY use tools:
                1. Call `list_agents()` to find available counselors
                2. Call `create_thread(threadName='urgent_crisis_support', participantIds=[counselor_ids, self_id])`
                3. Use Discord agent to notify: `send_message()` with "URGENT: High-risk caller needs immediate counselor support"
                4. Use `add_participant()` to bring counselor into thread
                5. Use `send_message()` to brief counselor with assessment summary

            **Moderate Risk** (Yes to Q2, No to Q3):
            - Say: "Thank you for sharing that with me. I'm going to connect you with a counselor who can help."
            - Use tools:
                1. Call `list_agents()` to find available counselors
                2. Call `create_thread(threadName='crisis_support', participantIds=[counselor_ids, self_id])`
                3. Use Discord agent to notify: "Caller with suicidal thoughts needs counselor support"
                4. Complete handoff process

            **Low Risk** (No to Q2 and Q3):
            - Say: "I'm glad you're safe right now. Let me connect you with a counselor to talk further."
            - Use standard tools for routine handoff

            **Uncertain/Evasive Responses**:
            - Say: "That's okay. I'm going to connect you with a counselor who can support you."
            - Treat as moderate risk for tool usage

            ### STEP 4: Counselor Handoff Process

            1. **Find Available Counselors:**
               ```
               Use list_agents() to get all connected agents
               Filter for counselor agents based on descriptions
               ```

            2. **Create Communication Thread:**
               ```
               Use create_thread(threadName='caller_support_[timestamp]', participantIds=[selected_counselor_id, self_id])
               ```

            3. **Notify Discord Channel:**
               ```
               Use Discord agent with send_message() to alert counselors:
               "New caller assessment complete - [Risk Level] - Counselor needed in thread [thread_id]"
               ```

            4. **Brief the Counselor:**
               ```
               Use send_message(threadId=thread_id, content="Assessment Summary: [Risk level], Caller responses: [Q1: answer, Q2: answer, Q3: answer], Additional context: [any relevant details]", mentions=[counselor_id])
               ```

            5. **Wait for Counselor Response:**
               ```
               Use wait_for_mentions(timeoutMs=60000) to confirm counselor availability
               If no response in 60 seconds, try another counselor
               ```

            6. **Complete Handoff:**
               - Tell caller: "I have [Counselor Name] joining us now. They are a trained counselor who will talk with you."
               - Step back and let counselor take over

            ## AVAILABLE CORAL TOOLS:
            {coral_tools_description}

            ## COMMUNICATION STYLE RULES:

            1. **Empathetic Language:**
               - "I hear you"
               - "Thank you for sharing that"
               - "That sounds really difficult"
               - "You're not alone"

            2. **Keep It Simple:**
               - Short sentences
               - No jargon or technical terms
               - Clear, gentle language

            3. **Sequential Conversation:**
               - Ask ONE question at a time
               - Wait for complete response
               - Acknowledge their answer
               - Then move to next step

            4. **Voice Interaction Adaptation:**
               - Speak slowly and clearly
               - Use appropriate pauses
               - Avoid bullet points or lists in speech
               - Express warmth through tone-appropriate language

            ## SAFETY BOUNDARIES:

            ❌ **Never Do:**
            - Give medical advice or therapy
            - Provide coping strategies or solutions
            - Promise confidentiality
            - Keep caller in extended conversation
            - Ask multiple questions at once
            - Skip the tool usage for counselor connection

            ✅ **Always Do:**
            - Follow the sequential question flow
            - Use tools to connect with counselors
            - Notify Discord channel of new callers
            - Brief counselors with assessment details
            - Hand off promptly after assessment

            ## EXAMPLE CONVERSATION FLOW:

            **User:** "I'm feeling really hopeless..."
            **Agent:** "Hi, thank you for reaching out. I hear that you're feeling hopeless right now, and I'm here to listen. I'd like to ask you a few questions so I can connect you with a counselor who can help. Are you safe right now?"

            **User:** "I guess so..."
            **Agent:** "Thank you for telling me that. Do you have thoughts of hurting yourself?"

            [Wait for response, then continue with next question]

            Remember: Your role is brief triage assessment followed by immediate tool-assisted handoff to trained counselors. The tools are essential for proper counselor notification and connection.
            """
        ),
        ("human", "{user_input}\n\nChat History:\n{chat_history}"),
        ("placeholder", "{agent_scratchpad}")
    ])

    model = init_chat_model(
        model=os.getenv("MODEL_NAME"),
        model_provider=os.getenv("MODEL_PROVIDER"),
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=float(os.getenv("MODEL_TEMPERATURE", DEFAULT_TEMPERATURE)),
        max_tokens=int(os.getenv("MODEL_MAX_TOKENS", DEFAULT_MAX_TOKENS)),
        base_url=os.getenv("MODEL_BASE_URL", None)
    )

    agent = create_tool_calling_agent(model, coral_tools, prompt)
    executor = AgentExecutor(agent=agent, tools=coral_tools, verbose=True, return_intermediate_steps=True)
    
    return executor

async def initialize_system():
    """Initialize the coral connection and agent"""
    global agent_executor, agent_tools, elevenlabs_client
    
    config = load_config()
    
    # Initialize ElevenLabs client for both STT and TTS
    elevenlabs_client = ElevenLabs(api_key=config["elevenlabs_api_key"])
    voice_logger.info("✅ ElevenLabs client initialized for speech-to-text and text-to-speech (Free Tier)")
    voice_logger.info(f"🗣️ Using voice: {config['elevenlabs_voice_id']}")
    voice_logger.info(f"🎯 Using model: {config['elevenlabs_model']}")
    
    # Check if pydub is available for MP3 handling
    if PYDUB_AVAILABLE:
        voice_logger.info("🎵 MP3 audio support enabled via pydub")
    else:
        voice_logger.warning("⚠️  MP3 audio will use silence fallback (install pydub for proper MP3 support)")

    coral_params = {
        "agentId": config["agent_id"],
        "agentDescription": "Voice-enabled suicide hotline triage assistant that provides initial safety assessment and connects callers to trained counselors"
    }
    
    query_string = urllib.parse.urlencode(coral_params)
    coral_server_url = f"{config['coral_sse_url']}?{query_string}"
    logger.info(f"Connecting to Coral Server: {coral_server_url}")

    timeout = float(os.getenv("TIMEOUT_MS", "30000"))
    
    client = MultiServerMCPClient(
        connections={
            "coral": {
                "transport": "sse",
                "url": coral_server_url,
                "timeout": timeout,
                "sse_read_timeout": timeout,
            }
        }
    )
    logger.info("Coral Server connection established")

    coral_tools = await client.get_tools(server_name="coral")
    logger.info(f"Retrieved {len(coral_tools)} coral tools")

    # Verify required tools
    required_coral_tools = [LIST_AGENTS_TOOL, CREATE_THREAD_TOOL, SEND_MESSAGE_TOOL, WAIT_FOR_MENTIONS_TOOL]
    available_tools = [tool.name for tool in coral_tools]
    
    for tool_name in required_coral_tools:
        if tool_name not in available_tools:
            error_message = f"Required coral tool '{tool_name}' not found"
            logger.error(error_message)
            raise ValueError(error_message)
    
    agent_tools = {tool.name: tool for tool in coral_tools}
    agent_executor = await create_agent(coral_tools)
    
    # Log initialization success with audio format support
    if PYDUB_AVAILABLE:
        logger.info("🎧 Voice-enabled triage agent system initialized with ElevenLabs + MP3 Support (pydub)")
    else:
        logger.info("🎧 Voice-enabled triage agent system initialized with ElevenLabs (Free Tier, MP3 fallback)")

async def text_interface_mode():
    """Run the text-based interface"""
    user_input_handler = UserInputHandler()
    input_thread = user_input_handler.start_input_thread()
    
    # Send initial greeting
    initial_greeting = "Crisis Support Line - I'm here to listen. Can you tell me what's going on today?"
    print(f"Triage Agent: {initial_greeting}")
    
    try:
        while user_input_handler.running:
            user_input = user_input_handler.get_input()
            
            if user_input is None:
                await asyncio.sleep(0.1)
                continue
            
            if user_input == "":  # Quit signal
                break
                
            response = await process_triage_input(user_input)
            print(f"Triage Agent: {response}")
            print()  # Add blank line for readability
            
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received, shutting down...")
    finally:
        user_input_handler.stop()
        print("Triage agent stopped. Take care.")

async def main():
    """Main function with argument parsing for different interface modes"""
    parser = argparse.ArgumentParser(description="Crisis Support Triage Agent - Voice and Text Enabled with MP3 Audio Support")
    parser.add_argument(
        "--voice",
        action="store_true",
        help="Launch with voice interface using FastRTC"
    )
    parser.add_argument(
        "--phone",
        action="store_true",
        help="Launch with FastRTC phone interface (get a temp phone number)"
    )
    parser.add_argument(
        "--text",
        action="store_true",
        help="Launch with text-only interface (default)"
    )
    
    args = parser.parse_args()
    
    # Default to text interface if no specific mode is chosen
    if not (args.voice or args.phone):
        args.text = True
    
    try:
        # Initialize the system
        await initialize_system()
        
        if args.voice or args.phone:
            # Voice interface mode
            voice_logger.info("🎧 Initializing voice-enabled crisis support system...")
            
            if PYDUB_AVAILABLE:
                voice_logger.info("🌟 Using ElevenLabs with full MP3 audio support (pydub)")
            else:
                voice_logger.info("🌟 Using ElevenLabs with MP3 fallback (install pydub for full support)")
                voice_logger.info("💡 Install with: pip install pydub")
            
            voice_logger.info("🎙️ Using ElevenLabs Scribe for speech-to-text transcription")
            voice_logger.info("🔊 Using ElevenLabs TTS with MP3 format handling")
            
            stream = create_voice_stream()
            voice_logger.info("🎧 Voice stream handler configured with MP3 support")
            
            if args.phone:
                voice_logger.info("📞 Launching with FastRTC phone interface...")
                voice_logger.info("📞 Callers can now reach the crisis support line via phone")
                stream.fastphone()
            else:
                voice_logger.info("🌈 Launching with voice-enabled web interface...")
                voice_logger.info("🎙️ Callers can now speak directly to the crisis support system")
                stream.ui.launch()
                
        elif args.text:
            # Text interface mode
            logger.info("💬 Launching text-based crisis support interface...")
            await text_interface_mode()
            
    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())