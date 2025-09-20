# 🆘 Crisis Support Multi-Agent System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Discord](https://img.shields.io/badge/Discord-Integration-7289da)](https://discord.com/)
[![ElevenLabs](https://img.shields.io/badge/ElevenLabs-Voice%20AI-00d4aa)](https://elevenlabs.io/)

> **A revolutionary AI-powered crisis support system that provides immediate triage, voice interactions, and seamless handoff to trained counselors through intelligent agent orchestration.**

## 🎯 Project Overview

Our Crisis Support Multi-Agent System transforms how mental health crisis intervention is delivered by combining:
- **Voice-enabled AI triage** using ElevenLabs speech-to-text and text-to-speech
- **Intelligent agent orchestration** with Coral framework
- **Real-time Discord integration** for counselor coordination
- **Sequential safety assessment** following crisis intervention best practices
- **Multi-modal interfaces** supporting both voice and text interactions

## ✨ Key Features

### 🎙️ Voice-First Crisis Support
- **Real-time voice processing** with ElevenLabs Scribe STT and TTS
- **Phone integration** via FastRTC for direct crisis line calls
- **MP3 audio support** with fallback mechanisms for reliability
- **Empathetic voice responses** using carefully selected voice models

### 🧠 Intelligent Triage System
- **Sequential safety assessment** following clinical protocols
- **Risk level determination** (High/Moderate/Low risk classification)
- **Evidence-based questioning** for suicide risk evaluation
- **Immediate escalation** for high-risk situations

### 🤝 Multi-Agent Orchestration
- **Coral framework integration** for agent communication
- **Dynamic counselor assignment** based on availability
- **Thread-based conversations** for organized support delivery
- **Real-time handoff coordination** between triage and counselors

### 💬 Discord Integration
- **Counselor notification system** via Discord bot
- **Real-time status updates** for crisis response teams
- **Channel-based coordination** for support staff
- **Automated alert system** for urgent cases

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Voice Input   │    │  Triage Agent    │    │ Discord Agent   │
│  (ElevenLabs)   │────▶│    (Main)       │────▶│  (Notifications)│
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │ Coral Framework  │
                       │  (Orchestration) │
                       └──────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │ Counselor Agents │
                       │   (Support)      │
                       └──────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key
- ElevenLabs API key
- Discord bot token (optional)
- Coral orchestration server

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd crisis-support-system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install pydub  # For MP3 audio support
   ```

3. **Environment setup**
   ```bash
   cp .sample_env .env
   # Edit .env with your API keys and configuration
   ```

4. **Configure environment variables**
   ```env
   # AI Model Configuration
   OPENAI_API_KEY=sk-proj-your_openai_api_key_here
   MODEL_PROVIDER=openai
   MODEL_NAME=gpt-4o-mini
   
   # ElevenLabs Configuration
   ELEVENLABS_API_KEY=sk_your_elevenlabs_api_key_here
   ELEVENLABS_VOICE_ID=JBFqnCBsd6RMkjVDRZzb  # George - calm, supportive
   
   # Coral Orchestration
   CORAL_SSE_URL=http://localhost:5555/devmode/exampleApplication/privkey/session1/sse
   
   # Discord Integration (Optional)
   DISCORD_BOT_TOKEN=your_discord_bot_token_here
   CHANNEL_ID=your_channel_id_here
   ```

### Running the System

#### Voice Interface (Recommended)
```bash
python call_main.py --voice
```

#### Phone Interface
```bash
python call_main.py --phone
```

#### Text Interface
```bash
python call_main.py --text
```

#### Discord Agent (Separate Terminal)
```bash
cd discord_agent
python discord_notify_agent.py
```

## 🎭 Usage Examples

### Voice Interaction Flow
```
🎙️ Caller speaks: "I'm feeling really hopeless and don't know what to do..."

🤖 AI Triage: "Hi, thank you for reaching out. I hear that you're feeling 
                hopeless right now, and I'm here to listen. I'd like to ask 
                you a few questions so I can connect you with a counselor 
                who can help. Are you safe right now?"

🎙️ Caller: "I guess so, but I've been thinking about hurting myself..."

🤖 AI Triage: "Thank you for telling me that. This sounds very serious. 
                I'm connecting you right now with a trained counselor."

💬 [System creates thread, notifies Discord, connects counselor]
```

### Risk Assessment Protocol
1. **Initial Contact**: Warm greeting and explanation
2. **Safety Check**: "Are you safe right now?"
3. **Suicidal Ideation**: "Do you have thoughts of hurting yourself?"
4. **Plan Assessment**: "Do you have a plan to act on those thoughts?"
5. **Risk Classification**: High/Moderate/Low risk determination
6. **Counselor Connection**: Immediate handoff with context

## 🛠️ Technical Stack

### Core Technologies
- **Python 3.8+** - Primary development language
- **LangChain** - AI agent framework and model integration
- **FastRTC** - Real-time voice communication
- **ElevenLabs** - Speech-to-text and text-to-speech
- **Discord.py** - Discord bot integration
- **Coral Framework** - Multi-agent orchestration

### AI Models
- **OpenAI GPT-4o-mini** - Primary language model
- **ElevenLabs Scribe** - Speech recognition
- **ElevenLabs TTS** - Voice synthesis with multiple voice options

### Audio Processing
- **Pydub** - MP3 audio format support
- **NumPy** - Audio data manipulation
- **Wave** - WAV file processing

## 📊 System Capabilities

### 🎯 Crisis Assessment
- **Evidence-based protocols** following suicide prevention guidelines
- **Sequential questioning** to minimize caller overwhelm
- **Risk stratification** for appropriate response levels
- **Context preservation** across conversation sessions

### 🔄 Agent Coordination
- **Dynamic thread creation** for counselor handoffs
- **Real-time availability checking** for counselor assignment
- **Automated briefing** of counselors with caller context
- **Escalation protocols** for high-risk situations

### 🎵 Voice Processing
- **Multi-format audio support** (WAV, MP3, PCM)
- **Free-tier ElevenLabs optimization** with fallback handling
- **Low-latency processing** for natural conversation flow
- **Emotional tone preservation** in voice synthesis

## 🔧 Configuration Options

### Voice Configuration
```env
ELEVENLABS_VOICE_ID=JBFqnCBsd6RMkjVDRZzb  # George - calm, supportive
ELEVENLABS_MODEL=eleven_multilingual_v2
```

**Available Voices:**
- **George** (JBFqnCBsd6RMkjVDRZzb) - Calm, supportive
- **Bella** (EXAVITQu4vr4xnSDxMaL) - Friendly, warm
- **Antoni** (ErXwobaYiN019PkySvjV) - Deep, authoritative

### Model Configuration
```env
MODEL_TEMPERATURE=0.7  # Balance between creativity and consistency
MODEL_MAX_TOKENS=4000  # Sufficient for detailed responses
TIMEOUT_MS=30000       # 30-second timeout for responses
```

## 🚨 Safety Features

### Built-in Safeguards
- **Immediate escalation** for high-risk assessments
- **No therapy or medical advice** - focuses on triage only
- **Professional boundaries** maintained throughout interaction
- **Counselor handoff protocols** for all cases

### Error Handling
- **Graceful failure modes** with fallback responses
- **Audio processing fallbacks** when MP3 decoding fails
- **Network resilience** with retry mechanisms
- **Logging and monitoring** for system health

## 📈 Performance Metrics

### Response Times
- **Voice processing**: < 2 seconds end-to-end
- **Triage assessment**: < 30 seconds per question
- **Counselor handoff**: < 60 seconds from completion
- **System availability**: 99.9% uptime target

### Scalability
- **Concurrent calls**: Supports multiple simultaneous sessions
- **Agent scaling**: Dynamic counselor pool management
- **Resource optimization**: Efficient memory and CPU usage

## 🤝 Contributing

We welcome contributions to improve crisis support capabilities:

1. **Fork the repository**
2. **Create a feature branch**
3. **Implement improvements** with proper testing
4. **Submit pull request** with detailed description

### Development Guidelines
- Follow crisis intervention best practices
- Maintain empathetic communication standards
- Ensure proper error handling and logging
- Test thoroughly with various scenarios

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Crisis intervention specialists** for protocol guidance
- **ElevenLabs** for voice AI technology
- **OpenAI** for language model capabilities
- **Discord** for real-time communication platform
- **Mental health professionals** for best practice insights

## 📞 Support

For technical support or crisis intervention resources:
- **Technical Issues**: Create a GitHub issue
- **Crisis Support**: Contact your local crisis hotline
- **Documentation**: Check the `/docs` directory

---

**⚠️ Important Notice**: This system is designed to supplement, not replace, professional crisis intervention services. Always ensure qualified mental health professionals are available for direct support.