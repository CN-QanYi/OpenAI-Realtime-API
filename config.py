"""
配置文件 - 存放所有服务配置参数
"""
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AudioConfig:
    """音频配置"""
    # OpenAI Realtime API 使用的采样率
    OPENAI_SAMPLE_RATE: int = 24000
    # Pipecat/STT 内部使用的采样率
    INTERNAL_SAMPLE_RATE: int = 16000
    # 声道数
    CHANNELS: int = 1
    # 音频格式
    SAMPLE_WIDTH: int = 2  # 16-bit PCM
    # 每帧音频的时长（毫秒）
    FRAME_DURATION_MS: int = 20


@dataclass
class VADConfig:
    """语音活动检测配置"""
    # VAD 类型: "server_vad" 或 None（手动模式）
    type: str = "server_vad"
    # 静音检测阈值（毫秒）
    silence_duration_ms: int = 500
    # VAD 灵敏度阈值
    threshold: float = 0.5
    # 语音前缀填充时长（毫秒）
    prefix_padding_ms: int = 300


@dataclass
class LLMConfig:
    """语言模型配置"""
    # 使用的模型
    model: str = "gpt-4o"
    # 默认系统提示词
    default_instructions: str = "你是一个有帮助的AI助手。请用简洁的语言回答问题。"
    # 最大输出 token 数
    max_tokens: int = 4096
    # 温度参数
    temperature: float = 0.7


@dataclass
class STTConfig:
    """语音转文字配置"""
    # STT 服务提供商: "deepgram", "whisper", "azure"
    provider: str = "deepgram"
    # Deepgram API Key
    api_key: str = field(default_factory=lambda: os.getenv("DEEPGRAM_API_KEY", ""))
    # 语言
    language: str = "zh-CN"
    # 模型
    model: str = "nova-2"


@dataclass
class TTSConfig:
    """文字转语音配置"""
    # TTS 服务提供商: "elevenlabs", "azure", "edge"
    provider: str = "elevenlabs"
    # ElevenLabs API Key
    api_key: str = field(default_factory=lambda: os.getenv("ELEVENLABS_API_KEY", ""))
    # 声音 ID
    voice_id: str = "21m00Tcm4TlvDq8ikWAM"  # Rachel
    # 声音稳定性
    stability: float = 0.5
    # 相似度增强
    similarity_boost: float = 0.75


@dataclass
class ServerConfig:
    """服务器配置"""
    # 主机地址
    host: str = "0.0.0.0"
    # 端口号
    port: int = 8000
    # WebSocket 端点路径
    ws_path: str = "/v1/realtime"
    # 是否启用调试模式
    debug: bool = True
    # OpenAI API Key（用于验证客户端请求）
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))


@dataclass
class Config:
    """主配置类"""
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    server: ServerConfig = field(default_factory=ServerConfig)


# 全局配置实例
config = Config()


def load_config_from_env():
    """从环境变量加载配置"""
    config.stt.api_key = os.getenv("DEEPGRAM_API_KEY", config.stt.api_key)
    config.tts.api_key = os.getenv("ELEVENLABS_API_KEY", config.tts.api_key)
    config.server.openai_api_key = os.getenv("OPENAI_API_KEY", config.server.openai_api_key)
    config.server.debug = os.getenv("DEBUG", "true").lower() == "true"
    return config
