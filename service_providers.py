"""
服务提供商工厂 - 根据配置创建对应的 STT/LLM/TTS 服务实例
"""
import os
import logging
from typing import Optional, Callable, Awaitable
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


# ==================== STT 服务抽象基类 ====================

class BaseSTTProvider(ABC):
    """STT 服务抽象基类"""
    
    @abstractmethod
    async def transcribe(self, audio_bytes: bytes) -> str:
        """将音频转换为文本"""
        pass


class DeepgramSTTProvider(BaseSTTProvider):
    """Deepgram STT 服务"""
    
    def __init__(self, api_key: str, model: str = "nova-2", language: str = "zh-CN"):
        self.api_key = api_key
        self.model = model
        self.language = language
        self._client = None
        
    async def _get_client(self):
        if self._client is None:
            try:
                from deepgram import DeepgramClient, PrerecordedOptions
                self._client = DeepgramClient(self.api_key)
            except ImportError:
                raise ImportError("请安装 deepgram-sdk: pip install deepgram-sdk")
        return self._client
    
    async def transcribe(self, audio_bytes: bytes) -> str:
        """使用 Deepgram 进行语音识别"""
        try:
            from deepgram import PrerecordedOptions
            
            client = await self._get_client()
            
            options = PrerecordedOptions(
                model=self.model,
                language=self.language,
                smart_format=True,
            )
            
            response = await client.listen.asyncrest.v("1").transcribe_file(
                {"buffer": audio_bytes, "mimetype": "audio/raw"},
                options
            )
            
            transcript = response.results.channels[0].alternatives[0].transcript
            logger.info(f"Deepgram 转录结果: {transcript}")
            return transcript
            
        except Exception as e:
            logger.error(f"Deepgram 转录错误: {e}")
            return ""


class OpenAIWhisperSTTProvider(BaseSTTProvider):
    """OpenAI Whisper STT 服务"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self._client = None
        
    async def _get_client(self):
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
            except ImportError:
                raise ImportError("请安装 openai: pip install openai")
        return self._client
    
    async def transcribe(self, audio_bytes: bytes) -> str:
        """使用 OpenAI Whisper 进行语音识别"""
        try:
            import io
            import wave
            
            client = await self._get_client()
            
            # 将 PCM 转换为 WAV 格式
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(audio_bytes)
            wav_buffer.seek(0)
            wav_buffer.name = "audio.wav"
            
            response = await client.audio.transcriptions.create(
                model="whisper-1",
                file=wav_buffer,
                language="zh"
            )
            
            transcript = response.text
            logger.info(f"OpenAI Whisper 转录结果: {transcript}")
            return transcript
            
        except Exception as e:
            logger.error(f"OpenAI Whisper 转录错误: {e}")
            return ""


class LocalWhisperSTTProvider(BaseSTTProvider):
    """本地 Whisper STT 服务"""
    
    def __init__(self, model: str = "base"):
        self.model_name = model
        self._model = None
        
    def _load_model(self):
        if self._model is None:
            try:
                import whisper  # type: ignore
                logger.info(f"加载本地 Whisper 模型: {self.model_name}")
                self._model = whisper.load_model(self.model_name)
            except ImportError:
                raise ImportError("请安装 openai-whisper: pip install openai-whisper")
        return self._model
    
    async def transcribe(self, audio_bytes: bytes) -> str:
        """使用本地 Whisper 进行语音识别"""
        try:
            import numpy as np
            import tempfile
            import wave
            import asyncio
            
            model = self._load_model()
            
            # 将音频写入临时文件
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                with wave.open(f.name, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(16000)
                    wav_file.writeframes(audio_bytes)
                temp_path = f.name
            
            # 在线程池中运行 Whisper
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                lambda: model.transcribe(temp_path, language="zh")
            )
            
            # 删除临时文件
            import os
            os.unlink(temp_path)
            
            transcript = result["text"].strip()
            logger.info(f"本地 Whisper 转录结果: {transcript}")
            return transcript
            
        except Exception as e:
            logger.error(f"本地 Whisper 转录错误: {e}")
            return ""


# ==================== LLM 服务抽象基类 ====================

class BaseLLMProvider(ABC):
    """LLM 服务抽象基类"""
    
    @abstractmethod
    async def generate_stream(
        self, 
        prompt: str, 
        system_prompt: str,
        on_chunk: Callable[[str], Awaitable[None]]
    ) -> str:
        """流式生成文本响应"""
        pass


class OpenAILLMProvider(BaseLLMProvider):
    """OpenAI LLM 服务"""
    
    def __init__(
        self, 
        api_key: str, 
        model: str = "gpt-4o",
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.7,
        max_tokens: int = 4096
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None
        self._conversation_history = []
        
    async def _get_client(self):
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
            except ImportError:
                raise ImportError("请安装 openai: pip install openai")
        return self._client
    
    def clear_history(self):
        """清空对话历史"""
        self._conversation_history = []
    
    async def generate_stream(
        self, 
        prompt: str, 
        system_prompt: str,
        on_chunk: Callable[[str], Awaitable[None]]
    ) -> str:
        """流式生成 OpenAI 响应"""
        try:
            client = await self._get_client()
            
            # 构建消息列表
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(self._conversation_history)
            messages.append({"role": "user", "content": prompt})
            
            # 添加到历史
            self._conversation_history.append({"role": "user", "content": prompt})
            
            full_response = ""
            
            stream = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    full_response += text
                    await on_chunk(text)
            
            # 添加助手响应到历史
            self._conversation_history.append({"role": "assistant", "content": full_response})
            
            logger.info(f"OpenAI 响应完成: {full_response[:100]}...")
            return full_response
            
        except Exception as e:
            logger.error(f"OpenAI 生成错误: {e}")
            return f"抱歉，我遇到了一些问题: {str(e)}"


class OllamaLLMProvider(BaseLLMProvider):
    """Ollama LLM 服务"""
    
    def __init__(
        self, 
        base_url: str = "http://localhost:11434",
        model: str = "llama3:8b",
        temperature: float = 0.7
    ):
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self._conversation_history = []
        
    def clear_history(self):
        """清空对话历史"""
        self._conversation_history = []
    
    async def generate_stream(
        self, 
        prompt: str, 
        system_prompt: str,
        on_chunk: Callable[[str], Awaitable[None]]
    ) -> str:
        """流式生成 Ollama 响应"""
        try:
            import aiohttp
            import json
            
            # 构建消息列表
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(self._conversation_history)
            messages.append({"role": "user", "content": prompt})
            
            # 添加到历史
            self._conversation_history.append({"role": "user", "content": prompt})
            
            full_response = ""
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "stream": True,
                        "options": {"temperature": self.temperature}
                    }
                ) as response:
                    async for line in response.content:
                        if line:
                            try:
                                data = json.loads(line.decode())
                                if "message" in data and "content" in data["message"]:
                                    text = data["message"]["content"]
                                    full_response += text
                                    await on_chunk(text)
                            except json.JSONDecodeError:
                                continue
            
            # 添加助手响应到历史
            self._conversation_history.append({"role": "assistant", "content": full_response})
            
            logger.info(f"Ollama 响应完成: {full_response[:100]}...")
            return full_response
            
        except Exception as e:
            logger.error(f"Ollama 生成错误: {e}")
            return f"抱歉，我遇到了一些问题: {str(e)}"


# ==================== TTS 服务抽象基类 ====================

class BaseTTSProvider(ABC):
    """TTS 服务抽象基类"""
    
    @abstractmethod
    async def synthesize_stream(
        self, 
        text: str,
        on_audio_chunk: Callable[[bytes], Awaitable[None]]
    ) -> bytes:
        """流式合成语音"""
        pass


class ElevenLabsTTSProvider(BaseTTSProvider):
    """ElevenLabs TTS 服务"""
    
    def __init__(
        self, 
        api_key: str, 
        voice_id: str = "21m00Tcm4TlvDq8ikWAM",
        model: str = "eleven_monolingual_v1"
    ):
        self.api_key = api_key
        self.voice_id = voice_id
        self.model = model
        
    async def synthesize_stream(
        self, 
        text: str,
        on_audio_chunk: Callable[[bytes], Awaitable[None]]
    ) -> bytes:
        """流式合成 ElevenLabs 语音"""
        try:
            import aiohttp
            
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/stream"
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": self.api_key
            }
            data = {
                "text": text,
                "model_id": self.model,
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.75
                }
            }
            
            full_audio = b""
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=data) as response:
                    if response.status == 200:
                        async for chunk in response.content.iter_chunked(4096):
                            full_audio += chunk
                            await on_audio_chunk(chunk)
                    else:
                        error = await response.text()
                        logger.error(f"ElevenLabs TTS 错误: {error}")
            
            logger.info(f"ElevenLabs TTS 完成: {len(full_audio)} bytes")
            return full_audio
            
        except Exception as e:
            logger.error(f"ElevenLabs TTS 错误: {e}")
            return b""


class EdgeTTSProvider(BaseTTSProvider):
    """Edge TTS 服务 (免费)"""
    
    def __init__(self, voice: str = "zh-CN-XiaoxiaoNeural"):
        self.voice = voice
        
    async def synthesize_stream(
        self, 
        text: str,
        on_audio_chunk: Callable[[bytes], Awaitable[None]]
    ) -> bytes:
        """流式合成 Edge TTS 语音"""
        try:
            import edge_tts
            import io
            
            communicate = edge_tts.Communicate(text, self.voice)
            
            full_audio = b""
            
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data = chunk["data"]
                    full_audio += audio_data
                    await on_audio_chunk(audio_data)
            
            logger.info(f"Edge TTS 完成: {len(full_audio)} bytes")
            return full_audio
            
        except ImportError:
            raise ImportError("请安装 edge-tts: pip install edge-tts")
        except Exception as e:
            logger.error(f"Edge TTS 错误: {e}")
            return b""


class OpenAITTSProvider(BaseTTSProvider):
    """OpenAI TTS 服务"""
    
    def __init__(
        self, 
        api_key: str, 
        voice: str = "alloy",
        model: str = "tts-1",
        base_url: str = "https://api.openai.com/v1"
    ):
        self.api_key = api_key
        self.voice = voice
        self.model = model
        self.base_url = base_url
        self._client = None
        
    async def _get_client(self):
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
            except ImportError:
                raise ImportError("请安装 openai: pip install openai")
        return self._client
    
    async def synthesize_stream(
        self, 
        text: str,
        on_audio_chunk: Callable[[bytes], Awaitable[None]]
    ) -> bytes:
        """流式合成 OpenAI TTS 语音"""
        try:
            client = await self._get_client()
            
            response = await client.audio.speech.create(
                model=self.model,
                voice=self.voice,
                input=text,
                response_format="pcm"  # 16-bit PCM
            )
            
            full_audio = response.content
            
            # 分块发送
            chunk_size = 4096
            for i in range(0, len(full_audio), chunk_size):
                chunk = full_audio[i:i+chunk_size]
                await on_audio_chunk(chunk)
            
            logger.info(f"OpenAI TTS 完成: {len(full_audio)} bytes")
            return full_audio
            
        except Exception as e:
            logger.error(f"OpenAI TTS 错误: {e}")
            return b""


# ==================== 服务工厂 ====================

class ServiceFactory:
    """服务工厂 - 根据配置创建服务实例"""
    
    @staticmethod
    def create_stt_provider(provider: str, **kwargs) -> BaseSTTProvider:
        """创建 STT 服务提供商"""
        providers = {
            "deepgram": lambda: DeepgramSTTProvider(
                api_key=kwargs.get("api_key", os.getenv("DEEPGRAM_API_KEY", "")),
                model=kwargs.get("model", os.getenv("DEEPGRAM_MODEL", "nova-2")),
                language=kwargs.get("language", os.getenv("DEEPGRAM_LANGUAGE", "zh-CN"))
            ),
            "openai_whisper": lambda: OpenAIWhisperSTTProvider(
                api_key=kwargs.get("api_key", os.getenv("OPENAI_API_KEY", "")),
                base_url=kwargs.get("base_url", os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
            ),
            "local_whisper": lambda: LocalWhisperSTTProvider(
                model=kwargs.get("model", os.getenv("WHISPER_MODEL", "base"))
            )
        }
        
        if provider not in providers:
            raise ValueError(f"未知的 STT 服务提供商: {provider}. 可选: {list(providers.keys())}")
        
        logger.info(f"创建 STT 服务: {provider}")
        return providers[provider]()
    
    @staticmethod
    def create_llm_provider(provider: str, **kwargs) -> BaseLLMProvider:
        """创建 LLM 服务提供商"""
        providers = {
            "openai": lambda: OpenAILLMProvider(
                api_key=kwargs.get("api_key", os.getenv("OPENAI_API_KEY", "")),
                model=kwargs.get("model", os.getenv("OPENAI_MODEL", "gpt-4o")),
                base_url=kwargs.get("base_url", os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")),
                temperature=float(kwargs.get("temperature", os.getenv("LLM_TEMPERATURE", "0.7"))),
                max_tokens=int(kwargs.get("max_tokens", os.getenv("LLM_MAX_TOKENS", "4096")))
            ),
            "ollama": lambda: OllamaLLMProvider(
                base_url=kwargs.get("base_url", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")),
                model=kwargs.get("model", os.getenv("OLLAMA_MODEL", "llama3:8b")),
                temperature=float(kwargs.get("temperature", os.getenv("LLM_TEMPERATURE", "0.7")))
            ),
            "siliconflow": lambda: OpenAILLMProvider(
                api_key=kwargs.get("api_key", os.getenv("SILICONFLOW_API_KEY", "")),
                model=kwargs.get("model", os.getenv("SILICONFLOW_MODEL", "Qwen/Qwen2.5-7B-Instruct")),
                base_url=kwargs.get("base_url", os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")),
                temperature=float(kwargs.get("temperature", os.getenv("LLM_TEMPERATURE", "0.7"))),
                max_tokens=int(kwargs.get("max_tokens", os.getenv("LLM_MAX_TOKENS", "4096")))
            )
        }
        
        if provider not in providers:
            raise ValueError(f"未知的 LLM 服务提供商: {provider}. 可选: {list(providers.keys())}")
        
        logger.info(f"创建 LLM 服务: {provider}")
        return providers[provider]()
    
    @staticmethod
    def create_tts_provider(provider: str, **kwargs) -> BaseTTSProvider:
        """创建 TTS 服务提供商"""
        providers = {
            "elevenlabs": lambda: ElevenLabsTTSProvider(
                api_key=kwargs.get("api_key", os.getenv("ELEVENLABS_API_KEY", "")),
                voice_id=kwargs.get("voice_id", os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")),
                model=kwargs.get("model", os.getenv("ELEVENLABS_MODEL", "eleven_monolingual_v1"))
            ),
            "edge_tts": lambda: EdgeTTSProvider(
                voice=kwargs.get("voice", os.getenv("EDGE_TTS_VOICE", "zh-CN-XiaoxiaoNeural"))
            ),
            "openai_tts": lambda: OpenAITTSProvider(
                api_key=kwargs.get("api_key", os.getenv("OPENAI_API_KEY", "")),
                voice=kwargs.get("voice", os.getenv("OPENAI_TTS_VOICE", "alloy")),
                model=kwargs.get("model", os.getenv("OPENAI_TTS_MODEL", "tts-1")),
                base_url=kwargs.get("base_url", os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
            )
        }
        
        if provider not in providers:
            raise ValueError(f"未知的 TTS 服务提供商: {provider}. 可选: {list(providers.keys())}")
        
        logger.info(f"创建 TTS 服务: {provider}")
        return providers[provider]()
