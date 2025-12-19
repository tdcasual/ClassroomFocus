"""
WebDAV client for uploading session files to remote storage.
"""
import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

# Lazy import to avoid import errors when webdavclient3 is not installed
_webdav_available = None
_Client = None


def _ensure_webdav():
    """Lazily import webdavclient3."""
    global _webdav_available, _Client
    if _webdav_available is None:
        try:
            from webdav3.client import Client
            _Client = Client
            _webdav_available = True
        except ImportError:
            _webdav_available = False
            logger.warning("webdavclient3 not installed. WebDAV support disabled.")
    return _webdav_available


class WebDAVConfig:
    """WebDAV configuration."""
    
    def __init__(
        self,
        enabled: bool = False,
        url: str = "",
        username: str = "",
        password: str = "",
        remote_path: str = "/classroom_focus",
        upload_video: bool = True,
        upload_audio: bool = True,
        upload_stats: bool = True,
        upload_transcript: bool = True,
        upload_all: bool = False,
    ):
        self.enabled = enabled
        self.url = url.rstrip("/") if url else ""
        self.username = username
        self.password = password
        self.remote_path = remote_path.rstrip("/") if remote_path else "/classroom_focus"
        self.upload_video = upload_video
        self.upload_audio = upload_audio
        self.upload_stats = upload_stats
        self.upload_transcript = upload_transcript
        self.upload_all = upload_all
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "url": self.url,
            "username": self.username,
            "password": self.password,
            "remote_path": self.remote_path,
            "upload_video": self.upload_video,
            "upload_audio": self.upload_audio,
            "upload_stats": self.upload_stats,
            "upload_transcript": self.upload_transcript,
            "upload_all": self.upload_all,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WebDAVConfig":
        if not data:
            return cls()
        return cls(
            enabled=bool(data.get("enabled", False)),
            url=str(data.get("url", "")),
            username=str(data.get("username", "")),
            password=str(data.get("password", "")),
            remote_path=str(data.get("remote_path", "/classroom_focus")),
            upload_video=bool(data.get("upload_video", True)),
            upload_audio=bool(data.get("upload_audio", True)),
            upload_stats=bool(data.get("upload_stats", True)),
            upload_transcript=bool(data.get("upload_transcript", True)),
            upload_all=bool(data.get("upload_all", False)),
        )
    
    def is_valid(self) -> bool:
        """Check if configuration has required fields."""
        return bool(self.url and self.username)


class WebDAVClient:
    """WebDAV client wrapper for uploading session files."""
    
    def __init__(self, config: WebDAVConfig):
        self.config = config
        self._client = None
    
    def _get_client(self):
        """Get or create WebDAV client."""
        if not _ensure_webdav():
            raise RuntimeError("webdavclient3 not installed")
        
        if self._client is None:
            options = {
                "webdav_hostname": self.config.url,
                "webdav_login": self.config.username,
                "webdav_password": self.config.password,
            }
            self._client = _Client(options)
        return self._client
    
    def test_connection(self) -> Dict[str, Any]:
        """Test WebDAV connection."""
        if not self.config.is_valid():
            return {"ok": False, "error": "Invalid configuration: URL and username required"}
        
        try:
            client = self._get_client()
            # Try to check if remote path exists
            exists = client.check(self.config.remote_path)
            if not exists:
                # Try to create the directory
                try:
                    client.mkdir(self.config.remote_path)
                    return {"ok": True, "message": f"Connected and created directory: {self.config.remote_path}"}
                except Exception as e:
                    return {"ok": True, "message": f"Connected but directory creation may have issues: {e}"}
            return {"ok": True, "message": f"Connected successfully. Remote path exists: {self.config.remote_path}"}
        except Exception as e:
            return {"ok": False, "error": str(e)}
    
    def ensure_remote_dir(self, remote_dir: str) -> bool:
        """Ensure remote directory exists, creating if necessary."""
        try:
            client = self._get_client()
            if not client.check(remote_dir):
                # Create parent directories recursively
                parts = remote_dir.strip("/").split("/")
                current = ""
                for part in parts:
                    current = f"{current}/{part}"
                    if not client.check(current):
                        client.mkdir(current)
            return True
        except Exception as e:
            logger.error(f"Failed to ensure remote directory {remote_dir}: {e}")
            return False
    
    def upload_file(self, local_path: str, remote_path: str) -> Dict[str, Any]:
        """Upload a single file to WebDAV."""
        try:
            client = self._get_client()
            
            # Ensure parent directory exists
            remote_dir = str(Path(remote_path).parent)
            if remote_dir and remote_dir != "/":
                self.ensure_remote_dir(remote_dir)
            
            client.upload_sync(remote_path=remote_path, local_path=local_path)
            return {"ok": True, "remote_path": remote_path}
        except Exception as e:
            logger.error(f"Failed to upload {local_path} to {remote_path}: {e}")
            return {"ok": False, "error": str(e)}
    
    def upload_session(self, session_dir: str, session_id: str) -> Dict[str, Any]:
        """Upload session files to WebDAV based on configuration."""
        if not self.config.enabled:
            return {"ok": False, "error": "WebDAV is disabled"}
        
        if not self.config.is_valid():
            return {"ok": False, "error": "Invalid WebDAV configuration"}
        
        session_path = Path(session_dir)
        if not session_path.exists():
            return {"ok": False, "error": f"Session directory not found: {session_dir}"}
        
        remote_session_dir = f"{self.config.remote_path}/{session_id}"
        
        # Determine which files to upload
        files_to_upload: List[str] = []
        
        if self.config.upload_all:
            # Upload all files
            files_to_upload = [f.name for f in session_path.iterdir() if f.is_file()]
        else:
            # Selective upload
            if self.config.upload_video:
                files_to_upload.extend(["session.mp4", "temp_video.avi"])
            if self.config.upload_audio:
                files_to_upload.extend(["temp_audio.wav"])
            if self.config.upload_stats:
                files_to_upload.extend(["stats.json", "lesson_summary.json", "cv_events.jsonl", "faces.jsonl"])
            if self.config.upload_transcript:
                files_to_upload.extend(["transcript.txt", "asr.jsonl"])
        
        # Upload files
        results = []
        uploaded_count = 0
        failed_count = 0
        
        for filename in files_to_upload:
            local_file = session_path / filename
            if not local_file.exists():
                continue
            
            remote_file = f"{remote_session_dir}/{filename}"
            result = self.upload_file(str(local_file), remote_file)
            
            if result.get("ok"):
                uploaded_count += 1
                results.append({"file": filename, "ok": True})
            else:
                failed_count += 1
                results.append({"file": filename, "ok": False, "error": result.get("error")})
        
        return {
            "ok": failed_count == 0,
            "uploaded": uploaded_count,
            "failed": failed_count,
            "results": results,
            "remote_dir": remote_session_dir,
        }
    
    def list_remote_sessions(self) -> Dict[str, Any]:
        """List sessions on remote WebDAV."""
        try:
            client = self._get_client()
            items = client.list(self.config.remote_path)
            # Filter out parent directory entries and the path itself
            # Items typically include the current dir as first item (trailing /)
            sessions = []
            remote_dir = self.config.remote_path.strip("/")
            for item in items:
                if not item:
                    continue
                item_clean = item.strip("/")
                # Skip if it's the remote_path itself or empty
                if not item_clean or item_clean == remote_dir:
                    continue
                # Only include the last segment (session folder name)
                parts = item_clean.split("/")
                session_name = parts[-1] if parts else ""
                if session_name:
                    sessions.append(session_name)
            return {"ok": True, "sessions": sessions}
        except Exception as e:
            return {"ok": False, "error": str(e)}


def create_client(config_dict: Dict[str, Any]) -> Optional[WebDAVClient]:
    """Create WebDAV client from config dictionary."""
    config = WebDAVConfig.from_dict(config_dict)
    if not config.enabled or not config.is_valid():
        return None
    return WebDAVClient(config)
