"""
音频工具 - 处理音频格式转换和重采样
支持 24kHz <-> 16kHz 的转换，用于 OpenAI 协议与内部处理之间的适配
"""
import io
import struct
import logging
from typing import Optional, Tuple, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# 尝试导入可选的高质量重采样库
try:
    import soxr  # type: ignore
    HAS_SOXR = True
    logger.info("使用 soxr 进行高质量音频重采样")
except ImportError:
    HAS_SOXR = False
    logger.info("soxr 不可用，使用 numpy 线性插值进行重采样")

try:
    import scipy.signal as signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class AudioResampler:
    """
    音频重采样器
    支持 PCM16 格式的采样率转换
    """
    
    def __init__(self, input_rate: int = 24000, output_rate: int = 16000):
        """
        初始化重采样器
        
        Args:
            input_rate: 输入采样率
            output_rate: 输出采样率
        """
        self.input_rate = input_rate
        self.output_rate = output_rate
        self.ratio = output_rate / input_rate
    
    def resample(self, audio_data: bytes) -> bytes:
        """
        重采样音频数据
        
        Args:
            audio_data: PCM16 格式的音频数据
            
        Returns:
            重采样后的 PCM16 音频数据
        """
        if self.input_rate == self.output_rate:
            return audio_data
        
        # 将 bytes 转换为 numpy 数组
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
        
        if len(audio_array) == 0:
            return audio_data
        
        # 执行重采样
        if HAS_SOXR:
            # 使用高质量的 soxr 重采样
            resampled = soxr.resample(
                audio_array, 
                self.input_rate, 
                self.output_rate,
                quality='HQ'
            )
        elif HAS_SCIPY:
            # 使用 scipy 重采样
            num_samples = int(len(audio_array) * self.ratio)
            resampled = signal.resample(audio_array, num_samples)
        else:
            # 使用简单的 numpy 线性插值
            resampled = self._numpy_resample(audio_array)
        
        # 转换回 int16 并返回 bytes
        resampled_array: NDArray[np.float32] = np.asarray(resampled, dtype=np.float32)
        clipped = np.clip(resampled_array, -32768, 32767).astype(np.int16)
        return clipped.tobytes()
    
    def _numpy_resample(self, audio_array: np.ndarray) -> np.ndarray:
        """使用 numpy 进行简单的线性插值重采样"""
        old_indices = np.arange(len(audio_array))
        new_length = int(len(audio_array) * self.ratio)
        new_indices = np.linspace(0, len(audio_array) - 1, new_length)
        return np.interp(new_indices, old_indices, audio_array)


class AudioConverter:
    """
    音频格式转换器
    处理 OpenAI Realtime API 与 Pipecat 之间的音频格式转换
    """
    
    # OpenAI Realtime API 默认使用 24kHz
    OPENAI_SAMPLE_RATE = 24000
    # Pipecat/STT 通常使用 16kHz
    INTERNAL_SAMPLE_RATE = 16000
    
    def __init__(self):
        # 输入转换：24kHz -> 16kHz（客户端音频 -> 内部处理）
        self.input_resampler = AudioResampler(
            input_rate=self.OPENAI_SAMPLE_RATE,
            output_rate=self.INTERNAL_SAMPLE_RATE
        )
        # 输出转换：16kHz -> 24kHz（内部处理 -> 客户端）
        self.output_resampler = AudioResampler(
            input_rate=self.INTERNAL_SAMPLE_RATE,
            output_rate=self.OPENAI_SAMPLE_RATE
        )
    
    def client_to_internal(self, audio_data: bytes) -> bytes:
        """
        将客户端音频转换为内部格式
        24kHz -> 16kHz
        """
        return self.input_resampler.resample(audio_data)
    
    def internal_to_client(self, audio_data: bytes) -> bytes:
        """
        将内部音频转换为客户端格式
        16kHz -> 24kHz
        """
        return self.output_resampler.resample(audio_data)


class AudioBuffer:
    """
    音频缓冲区
    用于累积和管理音频数据
    """
    
    def __init__(self, max_duration_seconds: float = 60.0, sample_rate: int = 24000):
        """
        初始化音频缓冲区
        
        Args:
            max_duration_seconds: 最大缓冲时长（秒）
            sample_rate: 采样率
        """
        self.sample_rate = sample_rate
        self.bytes_per_sample = 2  # PCM16
        self.max_bytes = int(max_duration_seconds * sample_rate * self.bytes_per_sample)
        self._buffer = io.BytesIO()
        self._total_bytes = 0
    
    def append(self, audio_data: bytes) -> None:
        """添加音频数据到缓冲区"""
        self._buffer.write(audio_data)
        self._total_bytes += len(audio_data)
        
        # 如果超过最大限制，删除最早的数据
        if self._total_bytes > self.max_bytes:
            self._trim_buffer()
    
    def _trim_buffer(self) -> None:
        """裁剪缓冲区，保留最新的数据"""
        current_data = self._buffer.getvalue()
        excess = self._total_bytes - self.max_bytes
        self._buffer = io.BytesIO(current_data[excess:])
        self._total_bytes = len(self._buffer.getvalue())
    
    def get_all(self) -> bytes:
        """获取所有缓冲的音频数据"""
        return self._buffer.getvalue()
    
    def clear(self) -> None:
        """清空缓冲区"""
        self._buffer = io.BytesIO()
        self._total_bytes = 0
    
    def get_duration_ms(self) -> int:
        """获取缓冲区中音频的时长（毫秒）"""
        samples = self._total_bytes // self.bytes_per_sample
        return int(samples / self.sample_rate * 1000)
    
    @property
    def size(self) -> int:
        """获取缓冲区大小（字节）"""
        return self._total_bytes


class AudioFrameProcessor:
    """
    音频帧处理器
    将连续的音频流分割成固定大小的帧
    """
    
    def __init__(self, frame_duration_ms: int = 20, sample_rate: int = 16000):
        """
        初始化帧处理器
        
        Args:
            frame_duration_ms: 每帧的时长（毫秒）
            sample_rate: 采样率
        """
        self.frame_duration_ms = frame_duration_ms
        self.sample_rate = sample_rate
        self.bytes_per_sample = 2  # PCM16
        self.frame_size = int(sample_rate * frame_duration_ms / 1000) * self.bytes_per_sample
        self._pending = b''
    
    def process(self, audio_data: bytes) -> list:
        """
        处理音频数据，返回完整的帧列表
        
        Args:
            audio_data: 输入音频数据
            
        Returns:
            完整帧的列表
        """
        data = self._pending + audio_data
        frames = []
        
        while len(data) >= self.frame_size:
            frames.append(data[:self.frame_size])
            data = data[self.frame_size:]
        
        self._pending = data
        return frames
    
    def flush(self) -> Optional[bytes]:
        """
        刷新剩余数据
        
        Returns:
            剩余的音频数据（可能不足一帧）
        """
        if self._pending:
            data = self._pending
            self._pending = b''
            return data
        return None


def calculate_audio_duration_ms(audio_bytes: bytes, sample_rate: int = 24000, 
                                bytes_per_sample: int = 2) -> int:
    """
    计算音频数据的时长（毫秒）
    
    Args:
        audio_bytes: 音频数据
        sample_rate: 采样率
        bytes_per_sample: 每个采样点的字节数
        
    Returns:
        时长（毫秒）
    """
    samples = len(audio_bytes) // bytes_per_sample
    return int(samples / sample_rate * 1000)


def pcm16_to_float32(audio_bytes: bytes) -> np.ndarray:
    """将 PCM16 音频转换为 float32 数组（范围 -1.0 到 1.0）"""
    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
    return audio_int16.astype(np.float32) / 32768.0


def float32_to_pcm16(audio_array: np.ndarray) -> bytes:
    """将 float32 数组转换为 PCM16 音频"""
    audio_int16 = (audio_array * 32768.0).clip(-32768, 32767).astype(np.int16)
    return audio_int16.tobytes()


def generate_silence(duration_ms: int, sample_rate: int = 24000) -> bytes:
    """
    生成静音音频
    
    Args:
        duration_ms: 时长（毫秒）
        sample_rate: 采样率
        
    Returns:
        PCM16 格式的静音数据
    """
    samples = int(sample_rate * duration_ms / 1000)
    return b'\x00\x00' * samples


# 全局音频转换器实例
audio_converter = AudioConverter()
