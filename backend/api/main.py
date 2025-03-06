import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import uuid

from pydantic.v1 import BaseSettings


# Configuration
class Settings(BaseSettings):
    RECORDING_PATH: Path = Path("./assets/trajectories")
    LOG_LEVEL: str = "INFO"

# Data models
class PositionData(BaseModel):
    timestamp: float
    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None
    rx: Optional[float] = None
    ry: Optional[float] = None
    rz: Optional[float] = None

class WebSocketMessage(BaseModel):
    action: str
    data: PositionData

class ConnectionManager:
    def __init__(self):
        self.spectators: Dict[uuid.UUID, WebSocket] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> uuid.UUID:
        await websocket.accept()
        spectator_id = uuid.uuid4()
        async with self._lock:
            self.spectators[spectator_id] = websocket
        return spectator_id

    async def disconnect(self, spectator_id: uuid.UUID) -> None:
        async with self._lock:
            self.spectators.pop(spectator_id, None)

    async def broadcast(self, message: dict) -> None:
        async with self._lock:
            await asyncio.gather(
                *[spectator.send_text(json.dumps(message))
                  for spectator in self.spectators.values()],
                return_exceptions=True
            )



# Application setup
app = FastAPI()
settings = Settings()
manager = ConnectionManager()

# Configure logging
logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

@app.websocket("/spectate")
async def process_spectator(websocket: WebSocket):
    spectator_id = await manager.connect(websocket)
    logger.info(f"Spectator {spectator_id} connected")

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await manager.disconnect(spectator_id)
        logger.info(f"Spectator {spectator_id} disconnected")


@app.websocket("/headset")
async def process_headset(websocket: WebSocket):
    await websocket.accept()
    recording: List[dict] = []
    is_recording = False
    start_time: Optional[int] = None

    try:
        while True:
            try:
                message = WebSocketMessage.model_validate_json(await websocket.receive_text())
                print(message)

                if message.action == "start_recording":
                    is_recording = True
                    start_time = message.data.timestamp
                    logger.info("Started recording")

                elif message.action == "stream_data":
                    await manager.broadcast(message.data.model_dump())

                    if is_recording:
                        recording.append(message.data.model_dump())

                elif message.action == "stop_recording" and is_recording:
                    is_recording = False
                    end_time = message.data.timestamp
                    duration = end_time - start_time if start_time else 0

                    logger.info(
                        f"Recording stats: size={len(recording)}, "
                        f"duration={duration:.1f}ms, "
                        f"frequency={1000 * len(recording) / duration:.1f}Hz"
                    )

                    # Save recording
                    recordings = list(settings.RECORDING_PATH.glob("recording_*.json"))
                    next_number = len(recordings) + 1

                    file_path = settings.RECORDING_PATH / f"recording_{next_number}.json"
                    file_path.write_text(json.dumps(recording))
                    logger.info(f"Saved recording to {file_path}")

                    recording = []

                else:
                    logger.warning(f"Unknown action received: {message.action}")

            except json.JSONDecodeError:
                logger.error("Received invalid JSON message")
            except Exception as e:
                logger.error(f"Error processing message: {str(e)}")
                break

    except WebSocketDisconnect:
        logger.info("Headset disconnected")