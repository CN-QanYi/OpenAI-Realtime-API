# OpenAI Realtime API 兼容服务器

一个完全复刻 OpenAI Realtime API 协议的本地 WebSocket 服务器，允许你使用本地或第三方模型替代 OpenAI。

## 🎯 项目目标

- **完全兼容**：对外复刻 OpenAI Realtime API 的协议（URL、JSON 事件格式、音频编码）
- **可替换后端**：对内使用 Pipecat 管道调用本地或第三方模型（Deepgram、Llama 3、ElevenLabs 等）
- **零客户端修改**：你的客户端应用只需修改 `baseUrl` 即可连接

## 📁 项目结构

```
├── main.py                 # FastAPI 主服务器
├── config.py               # 配置管理
├── protocol.py             # OpenAI Realtime API 协议定义
├── transport.py            # WebSocket Transport 层（协议翻译官）
├── pipeline_manager.py     # Pipecat 管道管理器
├── realtime_session.py     # 会话生命周期管理
├── audio_utils.py          # 音频处理工具（重采样等）
├── test_client.py          # 测试客户端
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
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple fastapi "uvicorn[standard]" websockets numpy scipy python-dotenv

# 或方法2: 直接安装（如果网络良好）
pip install -r requirements.txt
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

### 3. 连接客户端

将你的 OpenAI SDK 客户端的 `baseUrl` 修改为：
```
ws://localhost:8000/v1/realtime
```

### 4. 测试服务器

```bash
# 在新的终端窗口中，激活虚拟环境
.\.venv\Scripts\Activate.ps1

# 自动测试模式
python test_client.py

# 交互模式
python test_client.py -i
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

## ⚠️ 注意事项

### 音频采样率
- OpenAI 协议使用 **24kHz**
- 大多数 STT 模型使用 **16kHz**
- Transport 层自动处理重采样

### VAD 打断
检测到用户说话时，会发送 `input_audio_buffer.speech_started` 事件，客户端应清空本地音频缓冲区。

### JSON 格式严格性
`response_id` 和 `item_id` 字段必须存在，使用随机 UUID 填充。

## 🔜 后续计划

- [ ] 集成真实的 STT 服务（Deepgram/Whisper）
- [ ] 集成真实的 LLM 服务（OpenAI/Ollama）
- [ ] 集成真实的 TTS 服务（ElevenLabs/Edge TTS）
- [ ] 支持函数调用
- [ ] 支持多模态输入

## 📄 许可证

MIT License
