"""
会话管理器 - 管理单个 WebSocket 会话的完整生命周期
将 Transport 和 Pipeline 连接在一起
"""
import asyncio
import logging
from typing import Optional
from dataclasses import dataclass, field

from fastapi import WebSocket

from transport import OpenAIRealtimeTransport
from pipeline_manager import PipelineManager
from protocol import SessionConfig
from config import config

logger = logging.getLogger(__name__)


@dataclass
class SessionState:
    """会话状态"""
    session_id: str = ""
    is_active: bool = False
    current_response_id: Optional[str] = None
    current_item_id: Optional[str] = None
    total_input_tokens: int = 0
    total_output_tokens: int = 0


class RealtimeSession:
    """
    实时会话管理器
    
    协调 Transport 和 Pipeline 的交互：
    1. Transport 接收客户端事件，转换并传递给 Pipeline
    2. Pipeline 处理音频/文本，生成响应
    3. Transport 将响应转换为 OpenAI 格式发送给客户端
    """
    
    def __init__(self, websocket: WebSocket):
        """
        初始化会话
        
        Args:
            websocket: FastAPI WebSocket 连接
        """
        self.websocket = websocket
        self.transport = OpenAIRealtimeTransport(websocket)
        self.pipeline = PipelineManager()
        self.state = SessionState()
        
        self._setup_callbacks()
        
        logger.info("会话已创建")
    
    def _setup_callbacks(self):
        """设置各组件之间的回调连接"""
        
        # === Transport -> Pipeline 回调 ===
        
        # 音频帧回调：将音频数据推送到管道
        self.transport.on_audio_frame(self._on_audio_from_client)
        
        # 会话更新回调：更新管道配置
        self.transport.on_session_update(self._on_session_update)
        
        # 响应创建回调：强制触发响应生成
        self.transport.on_response_create(self._on_response_create)
        
        # 响应取消回调：取消当前响应
        self.transport.on_response_cancel(self._on_response_cancel)
        
        # === Pipeline -> Transport 回调 ===
        
        # 用户开始说话：发送打断信号
        self.pipeline.on_user_speech_start(self._on_user_speech_start)
        
        # 用户停止说话
        self.pipeline.on_user_speech_end(self._on_user_speech_end)
        
        # 转录完成
        self.pipeline.on_transcription(self._on_transcription)
        
        # 响应开始：创建响应对象
        self.pipeline.on_response_start(self._on_response_start)
        
        # 响应文本：发送文本增量
        self.pipeline.on_response_text(self._on_response_text)
        
        # 响应音频：发送音频增量
        self.pipeline.on_response_audio(self._on_response_audio)
        
        # 响应结束：完成响应
        self.pipeline.on_response_end(self._on_response_end)
    
    # ==================== 生命周期管理 ====================
    
    async def start(self):
        """启动会话"""
        self.state.is_active = True
        
        # 配置管道
        # 根据不同的 LLM 提供商选择模型名称
        if config.llm.provider == "openai":
            llm_model = config.llm.openai_model
        elif config.llm.provider == "ollama":
            llm_model = config.llm.ollama_model
        elif config.llm.provider == "siliconflow":
            llm_model = config.llm.siliconflow_model
        else:
            llm_model = "gpt-4o"  # 默认值
        
        self.pipeline.configure(
            vad_threshold=config.vad.threshold,
            vad_silence_ms=config.vad.silence_duration_ms,
            llm_model=llm_model,
            llm_instructions=config.llm.system_prompt,
            tts_voice="alloy"
        )
        
        # 启动管道
        await self.pipeline.start()
        
        # 启动 Transport（发送 session.created 事件）
        await self.transport.start()
        
        self.state.session_id = self.transport.state.session.id or ""
        
        logger.info(f"会话已启动: {self.state.session_id}")
    
    async def run(self):
        """运行会话主循环"""
        try:
            await self.transport.run()
        except Exception as e:
            logger.error(f"会话运行错误: {e}")
        finally:
            await self.stop()
    
    async def stop(self):
        """停止会话"""
        if not self.state.is_active:
            return
        
        self.state.is_active = False
        
        # 停止管道
        await self.pipeline.stop()
        
        # 关闭 Transport
        await self.transport.close()
        
        logger.info(f"会话已停止: {self.state.session_id}")
    
    # ==================== Transport -> Pipeline 回调实现 ====================
    
    async def _on_audio_from_client(self, audio_bytes: bytes):
        """处理来自客户端的音频"""
        await self.pipeline.push_audio(audio_bytes)
    
    async def _on_session_update(self, session: SessionConfig):
        """处理会话更新"""
        # 更新 LLM 指令
        if session.instructions:
            self.pipeline.update_instructions(session.instructions)
        
        # 更新 VAD 配置（Server VAD 始终启用）
        if session.turn_detection and self.pipeline.vad:
            self.pipeline.vad.threshold = session.turn_detection.threshold
            self.pipeline.vad.silence_duration_ms = session.turn_detection.silence_duration_ms
            logger.info(f"VAD 配置已更新: threshold={session.turn_detection.threshold}, silence={session.turn_detection.silence_duration_ms}ms")
        
        logger.info("会话配置已更新")
    
    async def _on_response_create(self):
        """处理响应创建请求"""
        # 强制触发响应生成
        await self.pipeline.force_response()
    
    async def _on_response_cancel(self):
        """处理响应取消"""
        await self.pipeline.cancel_response()
        await self.transport.cancel_response()
    
    # ==================== Pipeline -> Transport 回调实现 ====================
    
    async def _on_user_speech_start(self):
        """用户开始说话 - 发送打断信号"""
        # 如果正在生成响应，发送打断信号
        if self.state.current_response_id:
            await self.transport.send_speech_started()
            # 取消当前响应
            await self.transport.cancel_response()
            self.state.current_response_id = None
            self.state.current_item_id = None
        else:
            await self.transport.send_speech_started()
        
        logger.debug("用户开始说话")
    
    async def _on_user_speech_end(self):
        """用户停止说话"""
        await self.transport.send_speech_stopped()
        logger.debug("用户停止说话")
    
    async def _on_transcription(self, text: str):
        """转录完成"""
        logger.info(f"转录结果: {text}")
    
    async def _on_response_start(self):
        """响应开始 - 创建响应对象"""
        response_id, item_id = await self.transport.begin_response()
        self.state.current_response_id = response_id
        self.state.current_item_id = item_id
        logger.debug(f"响应开始: {response_id}")
    
    async def _on_response_text(self, text: str):
        """响应文本增量"""
        if self.state.current_response_id and self.state.current_item_id:
            await self.transport.send_transcript_delta(
                text,
                self.state.current_response_id,
                self.state.current_item_id
            )
    
    async def _on_response_audio(self, audio_bytes: bytes):
        """响应音频增量"""
        if self.state.current_response_id and self.state.current_item_id:
            await self.transport.send_audio_delta(
                audio_bytes,
                self.state.current_response_id,
                self.state.current_item_id
            )
    
    async def _on_response_end(self, full_text: str):
        """响应结束"""
        await self.transport.end_response(transcript=full_text)
        self.state.current_response_id = None
        self.state.current_item_id = None
        logger.debug("响应结束")


class SessionManager:
    """
    全局会话管理器
    管理所有活跃的会话
    """
    
    def __init__(self):
        self._sessions: dict[str, RealtimeSession] = {}
    
    async def create_session(self, websocket: WebSocket) -> RealtimeSession:
        """创建新会话"""
        session = RealtimeSession(websocket)
        await session.start()
        self._sessions[session.state.session_id] = session
        return session
    
    async def remove_session(self, session_id: str):
        """移除会话"""
        if session_id in self._sessions:
            session = self._sessions.pop(session_id)
            await session.stop()
    
    def get_session(self, session_id: str) -> Optional[RealtimeSession]:
        """获取会话"""
        return self._sessions.get(session_id)
    
    @property
    def active_count(self) -> int:
        """活跃会话数量"""
        return len(self._sessions)


# 全局会话管理器实例
session_manager = SessionManager()
