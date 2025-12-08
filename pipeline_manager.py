"""
管道管理器 - 管理 Pipecat 管道的创建和配置
提供一个简化的接口来构建语音处理管道
"""
import asyncio
import logging
from typing import Optional, Callable, Awaitable, List, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from config import config

logger = logging.getLogger(__name__)


# ==================== 帧类型定义 ====================
# 由于 Pipecat 可能未安装，我们定义自己的帧类型

@dataclass
class Frame:
    """基础帧类型"""
    pass


@dataclass
class AudioFrame(Frame):
    """音频帧"""
    audio: bytes
    sample_rate: int = 16000
    num_channels: int = 1


@dataclass
class InputAudioFrame(AudioFrame):
    """输入音频帧（来自用户）"""
    pass


@dataclass
class OutputAudioFrame(AudioFrame):
    """输出音频帧（发送给用户）"""
    pass


@dataclass
class TextFrame(Frame):
    """文本帧"""
    text: str


@dataclass
class TranscriptionFrame(TextFrame):
    """转录文本帧（STT 输出）"""
    pass


@dataclass
class LLMResponseFrame(TextFrame):
    """LLM 响应帧"""
    pass


@dataclass
class TTSAudioFrame(OutputAudioFrame):
    """TTS 输出音频帧"""
    pass


@dataclass
class UserStartedSpeakingFrame(Frame):
    """用户开始说话事件帧"""
    timestamp_ms: int = 0


@dataclass
class UserStoppedSpeakingFrame(Frame):
    """用户停止说话事件帧"""
    timestamp_ms: int = 0


@dataclass
class BotStartedSpeakingFrame(Frame):
    """机器人开始说话事件帧"""
    pass


@dataclass
class BotStoppedSpeakingFrame(Frame):
    """机器人停止说话事件帧"""
    pass


@dataclass 
class EndFrame(Frame):
    """结束帧，表示处理完成"""
    pass


# ==================== 服务抽象基类 ====================

class BaseService(ABC):
    """服务基类"""
    
    @abstractmethod
    async def process(self, frame: Frame) -> Optional[Frame]:
        """处理帧"""
        pass


class VADService(BaseService):
    """语音活动检测服务"""
    
    def __init__(self, threshold: float = 0.5, 
                 silence_duration_ms: int = 500,
                 prefix_padding_ms: int = 300):
        self.threshold = threshold
        self.silence_duration_ms = silence_duration_ms
        self.prefix_padding_ms = prefix_padding_ms
        self._is_speaking = False
        self._silence_frames = 0
        self._on_speech_start: Optional[Callable[[], Awaitable[None]]] = None
        self._on_speech_end: Optional[Callable[[], Awaitable[None]]] = None
    
    def on_speech_start(self, callback: Callable[[], Awaitable[None]]):
        """设置语音开始回调"""
        self._on_speech_start = callback
        return self
    
    def on_speech_end(self, callback: Callable[[], Awaitable[None]]):
        """设置语音结束回调"""
        self._on_speech_end = callback
        return self
    
    async def process(self, frame: Frame) -> Optional[Frame]:
        """处理音频帧，检测语音活动"""
        if not isinstance(frame, InputAudioFrame):
            return frame
        
        # 简单的能量检测 VAD（实际应使用 Silero VAD）
        import numpy as np
        audio_array = np.frombuffer(frame.audio, dtype=np.int16).astype(np.float32)
        
        if len(audio_array) == 0:
            return frame
        
        # 计算 RMS 能量
        rms = np.sqrt(np.mean(audio_array ** 2))
        is_speech = rms > (self.threshold * 3000)  # 阈值调整
        
        if is_speech and not self._is_speaking:
            self._is_speaking = True
            self._silence_frames = 0
            if self._on_speech_start:
                await self._on_speech_start()
            return UserStartedSpeakingFrame()
        
        elif not is_speech and self._is_speaking:
            self._silence_frames += 1
            # 根据静音时长判断是否结束
            frame_duration_ms = len(audio_array) / frame.sample_rate * 1000
            if self._silence_frames * frame_duration_ms > self.silence_duration_ms:
                self._is_speaking = False
                self._silence_frames = 0
                if self._on_speech_end:
                    await self._on_speech_end()
                return UserStoppedSpeakingFrame()
        
        return frame


class STTService(BaseService):
    """语音转文字服务（模拟）"""
    
    def __init__(self, language: str = "zh-CN"):
        self.language = language
        self._audio_buffer = b''
        self._on_transcription: Optional[Callable[[str], Awaitable[None]]] = None
    
    def on_transcription(self, callback: Callable[[str], Awaitable[None]]):
        """设置转录回调"""
        self._on_transcription = callback
        return self
    
    async def process(self, frame: Frame) -> Optional[Frame]:
        """处理音频帧，进行语音识别"""
        if isinstance(frame, UserStoppedSpeakingFrame):
            # 用户停止说话，处理累积的音频
            if self._audio_buffer:
                # 这里应该调用实际的 STT 服务
                # 目前返回模拟文本
                transcription = "[模拟转录文本] 你好，请问有什么可以帮助你的？"
                
                if self._on_transcription:
                    await self._on_transcription(transcription)
                
                self._audio_buffer = b''
                return TranscriptionFrame(text=transcription)
        
        elif isinstance(frame, InputAudioFrame):
            self._audio_buffer += frame.audio
        
        return frame


class LLMService(BaseService):
    """大语言模型服务（模拟）"""
    
    def __init__(self, model: str = "gpt-4o", 
                 instructions: str = "",
                 temperature: float = 0.7):
        self.model = model
        self.instructions = instructions
        self.temperature = temperature
        self._on_response_start: Optional[Callable[[], Awaitable[None]]] = None
        self._on_response_chunk: Optional[Callable[[str], Awaitable[None]]] = None
        self._on_response_end: Optional[Callable[[str], Awaitable[None]]] = None
    
    def on_response_start(self, callback: Callable[[], Awaitable[None]]):
        self._on_response_start = callback
        return self
    
    def on_response_chunk(self, callback: Callable[[str], Awaitable[None]]):
        self._on_response_chunk = callback
        return self
    
    def on_response_end(self, callback: Callable[[str], Awaitable[None]]):
        self._on_response_end = callback
        return self
    
    def update_instructions(self, instructions: str):
        """更新系统提示词"""
        self.instructions = instructions
    
    async def process(self, frame: Frame) -> Optional[Frame]:
        """处理转录文本，生成 LLM 响应"""
        if not isinstance(frame, TranscriptionFrame):
            return frame
        
        if self._on_response_start:
            await self._on_response_start()
        
        # 模拟 LLM 响应（实际应调用 OpenAI 或本地模型）
        response_text = f"你好！我收到了你的消息：「{frame.text}」。作为你的AI助手，我很乐意帮助你。请问有什么具体的问题吗？"
        
        # 模拟流式输出
        chunks = [response_text[i:i+10] for i in range(0, len(response_text), 10)]
        full_response = ""
        
        for chunk in chunks:
            full_response += chunk
            if self._on_response_chunk:
                await self._on_response_chunk(chunk)
            await asyncio.sleep(0.05)  # 模拟延迟
        
        if self._on_response_end:
            await self._on_response_end(full_response)
        
        return LLMResponseFrame(text=full_response)


class TTSService(BaseService):
    """文字转语音服务（模拟）"""
    
    def __init__(self, voice: str = "alloy", sample_rate: int = 16000):
        self.voice = voice
        self.sample_rate = sample_rate
        self._on_audio_chunk: Optional[Callable[[bytes], Awaitable[None]]] = None
        self._on_audio_end: Optional[Callable[[], Awaitable[None]]] = None
    
    def on_audio_chunk(self, callback: Callable[[bytes], Awaitable[None]]):
        self._on_audio_chunk = callback
        return self
    
    def on_audio_end(self, callback: Callable[[], Awaitable[None]]):
        self._on_audio_end = callback
        return self
    
    async def process(self, frame: Frame) -> Optional[Frame]:
        """处理 LLM 响应，生成语音"""
        if not isinstance(frame, LLMResponseFrame):
            return frame
        
        # 生成模拟音频（正弦波）
        import numpy as np
        
        # 根据文本长度生成相应时长的音频
        duration_ms = len(frame.text) * 80  # 每个字符约 80ms
        num_samples = int(self.sample_rate * duration_ms / 1000)
        
        # 生成 440Hz 正弦波作为模拟音频
        t = np.linspace(0, duration_ms / 1000, num_samples, dtype=np.float32)
        audio_float = np.sin(2 * np.pi * 440 * t) * 0.3
        audio_int16 = (audio_float * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        
        # 分块发送
        chunk_size = int(self.sample_rate * 0.1) * 2  # 100ms 的数据
        
        for i in range(0, len(audio_bytes), chunk_size):
            chunk = audio_bytes[i:i+chunk_size]
            if self._on_audio_chunk:
                await self._on_audio_chunk(chunk)
            await asyncio.sleep(0.05)
        
        if self._on_audio_end:
            await self._on_audio_end()
        
        return TTSAudioFrame(audio=audio_bytes, sample_rate=self.sample_rate)


# ==================== 管道管理器 ====================

class PipelineManager:
    """
    管道管理器
    协调 VAD -> STT -> LLM -> TTS 的处理流程
    """
    
    def __init__(self):
        self.vad: Optional[VADService] = None
        self.stt: Optional[STTService] = None
        self.llm: Optional[LLMService] = None
        self.tts: Optional[TTSService] = None
        
        # 回调函数
        self._on_user_speech_start: Optional[Callable[[], Awaitable[None]]] = None
        self._on_user_speech_end: Optional[Callable[[], Awaitable[None]]] = None
        self._on_transcription: Optional[Callable[[str], Awaitable[None]]] = None
        self._on_response_start: Optional[Callable[[], Awaitable[None]]] = None
        self._on_response_text: Optional[Callable[[str], Awaitable[None]]] = None
        self._on_response_audio: Optional[Callable[[bytes], Awaitable[None]]] = None
        self._on_response_end: Optional[Callable[[str], Awaitable[None]]] = None
        
        self._running = False
        self._audio_queue: asyncio.Queue = asyncio.Queue()
    
    def configure(self, 
                  vad_threshold: float = 0.5,
                  vad_silence_ms: int = 500,
                  llm_model: str = "gpt-4o",
                  llm_instructions: str = "",
                  tts_voice: str = "alloy") -> 'PipelineManager':
        """配置管道参数"""
        
        # 创建服务
        self.vad = VADService(
            threshold=vad_threshold,
            silence_duration_ms=vad_silence_ms
        )
        self.stt = STTService(language="zh-CN")
        self.llm = LLMService(
            model=llm_model,
            instructions=llm_instructions
        )
        self.tts = TTSService(voice=tts_voice)
        
        # 连接 VAD 回调
        self.vad.on_speech_start(self._handle_speech_start)
        self.vad.on_speech_end(self._handle_speech_end)
        
        # 连接 STT 回调
        self.stt.on_transcription(self._handle_transcription)
        
        # 连接 LLM 回调
        self.llm.on_response_start(self._handle_response_start)
        self.llm.on_response_chunk(self._handle_response_text)
        self.llm.on_response_end(self._handle_response_end)
        
        # 连接 TTS 回调
        self.tts.on_audio_chunk(self._handle_audio_chunk)
        
        logger.info("管道已配置")
        return self
    
    # ==================== 回调注册 ====================
    
    def on_user_speech_start(self, callback: Callable[[], Awaitable[None]]):
        self._on_user_speech_start = callback
        return self
    
    def on_user_speech_end(self, callback: Callable[[], Awaitable[None]]):
        self._on_user_speech_end = callback
        return self
    
    def on_transcription(self, callback: Callable[[str], Awaitable[None]]):
        self._on_transcription = callback
        return self
    
    def on_response_start(self, callback: Callable[[], Awaitable[None]]):
        self._on_response_start = callback
        return self
    
    def on_response_text(self, callback: Callable[[str], Awaitable[None]]):
        self._on_response_text = callback
        return self
    
    def on_response_audio(self, callback: Callable[[bytes], Awaitable[None]]):
        self._on_response_audio = callback
        return self
    
    def on_response_end(self, callback: Callable[[str], Awaitable[None]]):
        self._on_response_end = callback
        return self
    
    # ==================== 内部回调处理 ====================
    
    async def _handle_speech_start(self):
        if self._on_user_speech_start:
            await self._on_user_speech_start()
    
    async def _handle_speech_end(self):
        if self._on_user_speech_end:
            await self._on_user_speech_end()
    
    async def _handle_transcription(self, text: str):
        if self._on_transcription:
            await self._on_transcription(text)
    
    async def _handle_response_start(self):
        if self._on_response_start:
            await self._on_response_start()
    
    async def _handle_response_text(self, text: str):
        if self._on_response_text:
            await self._on_response_text(text)
    
    async def _handle_audio_chunk(self, audio: bytes):
        if self._on_response_audio:
            await self._on_response_audio(audio)
    
    async def _handle_response_end(self, full_text: str):
        if self._on_response_end:
            await self._on_response_end(full_text)
    
    # ==================== 公共接口 ====================
    
    async def start(self):
        """启动管道"""
        self._running = True
        logger.info("管道已启动")
    
    async def stop(self):
        """停止管道"""
        self._running = False
        logger.info("管道已停止")
    
    async def push_audio(self, audio_bytes: bytes):
        """推送音频数据到管道"""
        if not self._running:
            return
        
        frame = InputAudioFrame(audio=audio_bytes)
        
        # VAD 处理
        if self.vad:
            result = await self.vad.process(frame)
            
            # 如果检测到语音结束，触发 STT
            if isinstance(result, UserStoppedSpeakingFrame):
                if self.stt:
                    stt_result = await self.stt.process(result)
                    
                    # STT 完成后触发 LLM
                    if isinstance(stt_result, TranscriptionFrame) and self.llm:
                        llm_result = await self.llm.process(stt_result)
                        
                        # LLM 完成后触发 TTS
                        if isinstance(llm_result, LLMResponseFrame) and self.tts:
                            await self.tts.process(llm_result)
            
            # 继续收集音频用于 STT
            elif isinstance(result, (InputAudioFrame, UserStartedSpeakingFrame)):
                if self.stt:
                    await self.stt.process(frame)
    
    def update_instructions(self, instructions: str):
        """更新 LLM 系统提示词"""
        if self.llm:
            self.llm.update_instructions(instructions)
            logger.info("LLM 指令已更新")
    
    async def force_response(self):
        """强制生成响应（用于手动 VAD 模式）"""
        if self.stt:
            # 触发 STT 处理
            stt_result = await self.stt.process(UserStoppedSpeakingFrame())
            
            if isinstance(stt_result, TranscriptionFrame) and self.llm:
                llm_result = await self.llm.process(stt_result)
                
                if isinstance(llm_result, LLMResponseFrame) and self.tts:
                    await self.tts.process(llm_result)
    
    async def cancel_response(self):
        """取消当前响应"""
        # 清空待处理队列
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        logger.info("响应已取消")
