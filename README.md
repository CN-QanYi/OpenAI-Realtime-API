# OpenAI Realtime API 兼容服务器

一个完全复刻 OpenAI Realtime API 协议的本地 WebSocket 服务器，允许你使用本地或第三方模型替代 OpenAI。

## ✨ 特性

- 🔄 **完全兼容**：对外复刻 OpenAI Realtime API 的协议（URL、JSON 事件格式、音频编码）
- 🔌 **可替换后端**：对内使用 Pipecat 管道调用本地或第三方模型（Deepgram、Llama 3、ElevenLabs 等）
- 🚀 **零客户端修改**：你的客户端应用只需修改 `baseUrl` 即可连接
- 🎙️ **Push-to-Talk 客户端**：提供完整的终端 UI 客户端用于测试和演示
- 🎤 **Server VAD 支持**：集成 Pipecat 的 Silero VAD，实现自由麦模式的语音活动检测

## 📁 项目结构

```
├── main.py                 # FastAPI 主服务器
├── config.py               # 配置管理
├── protocol.py             # OpenAI Realtime API 协议定义
├── transport.py            # WebSocket Transport 层（协议翻译官）
├── pipeline_manager.py     # Pipecat 管道管理器
├── realtime_session.py     # 会话生命周期管理
├── audio_utils.py          # 音频处理工具（重采样、音频播放等）
├── push_to_talk_app.py     # Push-to-Talk 终端客户端（推荐）
├── test_client.py          # 简单测试客户端
└── requirements.txt        # 依赖列表
```

## 🚀 快速开始

### 1. 安装依赖

```bash
# 方法1: 创建虚拟环境（推荐）
python -m venv .venv

# 激活虚拟环境
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Windows CMD:
.venv\Scripts\activate.bat
# Linux/Mac:
source .venv/bin/activate

# 安装依赖（使用清华镜像源，速度更快）
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# 或方法2: 直接安装（如果网络良好）
pip install -r requirements.txt

# 核心依赖包括:
# - fastapi, uvicorn: Web 框架和服务器
# - websockets: WebSocket 支持
# - numpy, scipy: 音频处理
# - pipecat-ai: 提供 Server VAD (Silero VAD) 等音频处理功能
```

### 2. 启动服务器

```bash
# 确保已激活虚拟环境
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# 开发模式（自动重载）
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 或直接运行
python main.py
```

### 3. 运行客户端测试

#### 方式 1: Push-to-Talk 客户端（推荐）

提供完整的终端 UI 界面，支持按键说话功能：

```bash
# 安装客户端依赖
pip install textual sounddevice

# 在新的终端窗口中运行客户端
python push_to_talk_app.py
```

**使用说明：**
- 按 **K** 键开始录音，再次按 **K** 停止录音
- 按 **Q** 键退出应用
- 客户端会自动连接到 `ws://localhost:8000/v1/realtime`
- 可以在 `push_to_talk_app.py` 中设置 `USE_LOCAL_SERVER = False` 切换到 OpenAI 官方 API

**功能特性：**
- ✅ 实时显示会话 ID
- ✅ 录音状态指示器
- ✅ 实时显示 AI 响应文本
- ✅ 自动播放 AI 语音响应
- ✅ 完整的 TUI（终端用户界面）

#### 方式 2: 简单测试客户端

```bash
# 自动测试模式
python test_client.py

# 交互模式
python test_client.py -i
```

#### 方式 3: 使用 OpenAI SDK

在你的客户端代码中：

```python
from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy-key"  # 本地服务器不需要真实 key
)

async with client.realtime.connect(model="gpt-realtime") as conn:
    # 你的代码...
```

## 🔧 架构设计

### 数据流向

```
客户端 → OpenAI 格式 JSON → Transport (翻译) → Pipecat Pipeline
                                                    ↓
客户端 ← OpenAI 格式 JSON ← Transport (翻译) ← (VAD→STT→LLM→TTS)
```

### 核心组件

1. **Transport 层** (`transport.py`)
   - 接收 OpenAI 格式的客户端事件
   - 转换为 Pipecat 内部帧格式
   - 将输出转换回 OpenAI 格式

2. **Pipeline 管理器** (`pipeline_manager.py`)
   - VAD：语音活动检测
   - STT：语音转文字
   - LLM：语言模型推理
   - TTS：文字转语音

3. **会话管理** (`realtime_session.py`)
   - 管理 WebSocket 会话生命周期
   - 协调 Transport 和 Pipeline

4. **音频处理** (`audio_utils.py`)
   - 音频重采样（24kHz ↔ 16kHz）
   - 音频缓冲区管理
   - 异步音频播放器（用于客户端）

## 📋 支持的事件

### 客户端 → 服务器

| 事件类型 | 描述 |
|---------|------|
| `session.update` | 更新会话配置（VAD、指令等） |
| `input_audio_buffer.append` | 追加音频数据 |
| `input_audio_buffer.commit` | 提交音频缓冲区 |
| `input_audio_buffer.clear` | 清空音频缓冲区 |
| `conversation.item.create` | 创建对话项 |
| `response.create` | 请求生成响应 |
| `response.cancel` | 取消当前响应 |

### 服务器 → 客户端

| 事件类型 | 描述 |
|---------|------|
| `session.created` | 会话已创建 |
| `session.updated` | 会话已更新 |
| `input_audio_buffer.speech_started` | 检测到语音开始 |
| `input_audio_buffer.speech_stopped` | 检测到语音停止 |
| `response.created` | 响应已创建 |
| `response.audio.delta` | 音频增量 |
| `response.audio_transcript.delta` | 转录增量 |
| `response.done` | 响应完成 |

## ⚙️ 配置说明

配置文件位于 `config.py`，可以通过修改该文件来自定义服务器行为：

### 音频配置
```python
OPENAI_SAMPLE_RATE = 24000  # OpenAI 协议采样率
INTERNAL_SAMPLE_RATE = 16000  # 内部处理采样率
```

### VAD 配置（自由麦模式）
```python
# VAD 类型: "server_vad" (自由麦) 或 None (按键说话)
type = "server_vad"
# 静音检测时长（毫秒），超过此时长认为用户停止说话
silence_duration_ms = 500
# VAD 灵敏度阈值 (0.0-1.0)，越高越不敏感
threshold = 0.5
# 语音前缀填充时长（毫秒），保留语音开始前的音频
prefix_padding_ms = 300
```

**注意**: 自由麦模式需要 Pipecat 提供的 Silero VAD 支持，已包含在 `pipecat-ai` 依赖中。

### 模型配置
```python
# LLM 配置
model = "gpt-4o"  # 可替换为本地模型
default_instructions = "你是一个有帮助的AI助手。"

# STT 配置
provider = "deepgram"  # 可选: whisper, azure
api_key = os.getenv("DEEPGRAM_API_KEY", "")

# TTS 配置
provider = "elevenlabs"  # 可选: azure, edge
api_key = os.getenv("ELEVENLABS_API_KEY", "")
```

### 服务器配置
```python
host = "0.0.0.0"
port = 8000
ws_path = "/v1/realtime"
debug = True
```

## ⚠️ 注意事项

### 音频采样率
- OpenAI 协议使用 **24kHz**
- 大多数 STT 模型使用 **16kHz**
- `audio_utils.py` 自动处理重采样

### VAD 模式

#### Server VAD 模式（自由麦）
当配置 `turn_detection.type = "server_vad"` 时，服务器会使用 Pipecat 的 Silero VAD 自动检测用户的语音活动：
- 检测到用户开始说话时，发送 `input_audio_buffer.speech_started` 事件
- 检测到用户停止说话时，发送 `input_audio_buffer.speech_stopped` 事件
- 客户端收到 `speech_started` 事件后应立即停止播放 AI 音频，实现打断功能

#### 手动模式（按键说话）
当配置 `turn_detection = null` 时，需要客户端手动发送 `input_audio_buffer.commit` 来提交音频。

### JSON 格式严格性
`response_id` 和 `item_id` 字段必须存在，使用随机 UUID 填充。

### 客户端音频设备
`push_to_talk_app.py` 需要麦克风和扬声器支持。如遇问题：
- Windows: 确保安装了 `sounddevice` 包
- Mac: 需要 `brew install portaudio`
- Linux: 需要安装 `portaudio19-dev` 和 `python3-pyaudio`

## 🎬 快速演示

### 1. 启动服务器（终端 1）
```bash
python main.py
```
输出：
```
OpenAI Realtime API 兼容服务器启动
WebSocket 端点: ws://localhost:8000/v1/realtime
```

### 2. 启动客户端（终端 2）
```bash
python push_to_talk_app.py
```

### 3. 开始对话
1. 按 **K** 键开始录音
2. 说话（例如："你好，今天天气怎么样？"）
3. 再按 **K** 键停止录音
4. 等待 AI 响应（文本会显示在界面上，音频会自动播放）

## 🔄 替换为本地/第三方模型

### 修改配置文件

编辑 `config.py` 来使用本地或第三方服务：

#### 1. 使用本地 Whisper（STT）
```python
@dataclass
class STTConfig:
    provider: str = "whisper"  # 改为 whisper
    model: str = "base"  # 或 small, medium, large
    language: str = "zh"
```

#### 2. 使用 Ollama（LLM）
```python
@dataclass
class LLMConfig:
    model: str = "llama3:8b"  # 或其他 Ollama 模型
    api_base: str = "http://localhost:11434"  # Ollama 地址
    temperature: float = 0.7
```

#### 3. 使用 Edge TTS（免费 TTS）
```python
@dataclass
class TTSConfig:
    provider: str = "edge"  # 改为 edge（无需 API key）
    voice_id: str = "zh-CN-XiaoxiaoNeural"  # 中文语音
```

### 实现自定义后端

在 `pipeline_manager.py` 中实现你的自定义处理逻辑：

```python
class PipelineManager:
    async def process_audio(self, audio_bytes: bytes):
        # 1. STT: 将音频转为文本
        text = await your_stt_service.transcribe(audio_bytes)
        
        # 2. LLM: 生成回复
        response = await your_llm_service.generate(text)
        
        # 3. TTS: 将文本转为语音
        audio = await your_tts_service.synthesize(response)
        
        return audio, response
```

## 🐛 故障排除

### 客户端无法连接
```bash
# 检查服务器是否运行
curl http://localhost:8000/health

# 应返回: {"status":"healthy","active_sessions":0}
```

### 音频设备问题
```python
# 测试音频设备
python -c "import sounddevice as sd; print(sd.query_devices())"
```

### 模块导入错误
```bash
# 重新安装依赖
pip install --upgrade -r requirements.txt
```

### WebSocket 连接断开
- 检查防火墙设置
- 确保端口 8000 未被占用
- 查看服务器日志以获取详细错误信息

## 📊 性能优化

### 音频重采样
- 安装 `soxr` 以获得更高质量的重采样：
```bash
pip install soxr
```

### 减少延迟
- 调整 `config.py` 中的 VAD 参数
- 使用更快的本地模型
- 减小音频缓冲区大小

## 🔜 后续计划

- [x] 完整的协议实现
- [x] Push-to-Talk 终端客户端
- [x] 音频处理和重采样
- [x] 集成 Pipecat 提供的 Server VAD (Silero VAD)
- [ ] 集成真实的 STT 服务（Deepgram/Whisper）
- [ ] 集成真实的 LLM 服务（OpenAI/Ollama）
- [ ] 集成真实的 TTS 服务（ElevenLabs/Edge TTS）
- [ ] 支持函数调用
- [ ] 支持多模态输入
- [ ] Docker 部署支持

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

### 开发指南
1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

MIT License
