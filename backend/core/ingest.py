"""
Data ingestion module for Relation Radar.

This module handles converting raw text input into structured Event records,
storing them in the database, and indexing them in the vector store.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.models import Event  # noqa: E402
from backend.core.repositories import EventRepository  # noqa: E402
from backend.rag.retriever import get_retriever  # noqa: E402


@dataclass
class EventDraft:
    """
    Draft event extracted from raw text.
    
    This is used as an intermediate representation between
    raw text and the final Event model. Future AI extraction
    will produce a list of these drafts from a single text input.
    """
    
    summary: str
    raw_text: str
    occurred_at: Optional[str] = None
    raw_time_text: Optional[str] = None
    event_type: Optional[str] = None
    emotion: Optional[str] = None
    preferences: List[str] = field(default_factory=list)
    taboos: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    
    def to_event(self, person_ids: List[int]) -> Event:
        """Convert draft to Event model."""
        return Event(
            person_ids=person_ids,
            occurred_at=self.occurred_at,
            raw_time_text=self.raw_time_text,
            event_type=self.event_type,
            summary=self.summary,
            raw_text=self.raw_text,
            emotion=self.emotion,
            preferences=self.preferences,
            taboos=self.taboos,
            tags=self.tags,
        )


class TextExtractor:
    """
    Simple rule-based text extractor for event information.
    
    This is a placeholder implementation that will be replaced
    with AI-powered extraction (e.g., Qwen) in future versions.
    """
    
    # Time patterns
    TIME_PATTERNS = {
        r'今天': 'today',
        r'昨天': 'yesterday',
        r'前天': 'day_before_yesterday',
        r'上周': 'last_week',
        r'上个月': 'last_month',
        r'刚才': 'just_now',
        r'今晚': 'tonight',
        r'今早': 'this_morning',
        r'中午': 'noon',
        r'晚上': 'evening',
    }
    
    # Emotion patterns
    EMOTION_PATTERNS = {
        '开心': ['开心', '高兴', '愉快', '快乐', '兴奋', '满足', '幸福'],
        '悲伤': ['伤心', '难过', '悲伤', '哭', '失落'],
        '焦虑': ['焦虑', '担心', '紧张', '压力', '烦恼', '忧虑'],
        '生气': ['生气', '愤怒', '烦躁', '不满', '抱怨'],
        '平静': ['平静', '淡定', '轻松', '放松'],
    }
    
    # Event type patterns
    EVENT_TYPE_PATTERNS = {
        '聚餐': ['吃饭', '聚餐', '饭局', '吃火锅', '吃烧烤', '吃日料', '吃川菜', '喝酒', '餐厅'],
        '健身': ['健身', '运动', '锻炼', '跑步', '游泳', '健身房', '瑜伽'],
        '聊天': ['聊天', '聊', '说', '谈', '沟通', '交流', '讨论'],
        '约会': ['约会', '看电影', '逛街', '购物'],
        '工作': ['工作', '加班', '开会', '出差', '项目'],
        '生日': ['生日', '庆祝', '蛋糕'],
        '旅行': ['旅行', '旅游', '出游', '玩'],
    }
    
    # Preference patterns
    PREFERENCE_KEYWORDS = ['喜欢', '爱', '偏好', '喜爱', '钟爱', '热爱', '最爱']
    
    # Taboo patterns
    TABOO_KEYWORDS = ['不喜欢', '讨厌', '忌讳', '反感', '害怕', '不爱', '不吃', '不想']
    
    def extract(self, text: str) -> EventDraft:
        """
        Extract event information from raw text.
        
        Args:
            text: Raw text input
            
        Returns:
            EventDraft with extracted information
        """
        # Extract time
        raw_time_text, occurred_at = self._extract_time(text)
        
        # Extract emotion
        emotion = self._extract_emotion(text)
        
        # Extract event type
        event_type = self._extract_event_type(text)
        
        # Extract preferences and taboos
        preferences = self._extract_preferences(text)
        taboos = self._extract_taboos(text)
        
        # Generate tags
        tags = self._generate_tags(text, event_type, emotion)
        
        # Create summary (use full text for now, AI will generate better summaries)
        summary = text.strip()
        if len(summary) > 200:
            summary = summary[:200] + "..."
        
        return EventDraft(
            summary=summary,
            raw_text=text,
            occurred_at=occurred_at,
            raw_time_text=raw_time_text,
            event_type=event_type,
            emotion=emotion,
            preferences=preferences,
            taboos=taboos,
            tags=tags,
        )
    
    def _extract_time(self, text: str) -> tuple[Optional[str], Optional[str]]:
        """Extract time information from text."""
        raw_time_text = None
        occurred_at = None
        
        for pattern, time_type in self.TIME_PATTERNS.items():
            if re.search(pattern, text):
                raw_time_text = pattern
                # Convert to ISO format based on current date
                occurred_at = self._time_to_iso(time_type)
                break
        
        return raw_time_text, occurred_at
    
    def _time_to_iso(self, time_type: str) -> str:
        """Convert time type to ISO format string."""
        now = datetime.now()
        
        if time_type == 'today':
            return now.strftime("%Y-%m-%dT%H:%M:%S")
        elif time_type == 'yesterday':
            from datetime import timedelta
            yesterday = now - timedelta(days=1)
            return yesterday.strftime("%Y-%m-%dT%H:%M:%S")
        elif time_type == 'day_before_yesterday':
            from datetime import timedelta
            day = now - timedelta(days=2)
            return day.strftime("%Y-%m-%dT%H:%M:%S")
        elif time_type == 'last_week':
            from datetime import timedelta
            day = now - timedelta(weeks=1)
            return day.strftime("%Y-%m-%dT%H:%M:%S")
        elif time_type == 'last_month':
            from datetime import timedelta
            day = now - timedelta(days=30)
            return day.strftime("%Y-%m-%dT%H:%M:%S")
        else:
            return now.strftime("%Y-%m-%dT%H:%M:%S")
    
    def _extract_emotion(self, text: str) -> Optional[str]:
        """Extract emotion from text."""
        for emotion, keywords in self.EMOTION_PATTERNS.items():
            for keyword in keywords:
                if keyword in text:
                    return emotion
        return None
    
    def _extract_event_type(self, text: str) -> Optional[str]:
        """Extract event type from text."""
        for event_type, keywords in self.EVENT_TYPE_PATTERNS.items():
            for keyword in keywords:
                if keyword in text:
                    return event_type
        return None
    
    def _extract_preferences(self, text: str) -> List[str]:
        """Extract preferences from text."""
        preferences = []
        
        for keyword in self.PREFERENCE_KEYWORDS:
            # Find what comes after the keyword
            pattern = f'{keyword}(.{{1,20}}?)(?:[，。！？,.]|$)'
            matches = re.findall(pattern, text)
            for match in matches:
                pref = match.strip()
                if pref and len(pref) > 1:
                    preferences.append(f"{keyword}{pref}")
        
        return preferences
    
    def _extract_taboos(self, text: str) -> List[str]:
        """Extract taboos from text."""
        taboos = []
        
        for keyword in self.TABOO_KEYWORDS:
            pattern = f'{keyword}(.{{1,20}}?)(?:[，。！？,.]|$)'
            matches = re.findall(pattern, text)
            for match in matches:
                taboo = match.strip()
                if taboo and len(taboo) > 1:
                    taboos.append(f"{keyword}{taboo}")
        
        return taboos
    
    def _generate_tags(
        self,
        text: str,
        event_type: Optional[str],
        emotion: Optional[str]
    ) -> List[str]:
        """Generate tags for the event."""
        tags = []
        
        if event_type:
            tags.append(event_type)
        
        if emotion:
            tags.append(emotion)
        
        # Add common keywords as tags
        common_tags = ['工作', '生活', '朋友', '家人', '健康']
        for tag in common_tags:
            if tag in text and tag not in tags:
                tags.append(tag)
        
        return tags


def extract_events(text: str) -> List[EventDraft]:
    """
    Extract events from raw text using AI (Qwen) or fallback to rules.
    
    This function uses Qwen LLM to intelligently extract multiple events
    from a single text input, with structured JSON output.
    
    Args:
        text: Raw text input
        
    Returns:
        List of EventDraft objects extracted from the text
    """
    # Import here to avoid circular imports
    from backend.llm.local_client import get_llm_client
    
    client = get_llm_client()
    
    # Check if using Qwen mode for intelligent extraction
    if hasattr(client, '_mode') and client._mode == 'qwen':
        try:
            return _extract_events_with_qwen(text, client)
        except Exception as e:
            print(f"⚠️  Qwen extraction failed: {e}, falling back to rule-based extraction")
    
    # Fallback to rule-based extraction
    extractor = TextExtractor()
    draft = extractor.extract(text)
    return [draft]


def _extract_events_with_qwen(text: str, client) -> List[EventDraft]:
    """
    Use Qwen to extract structured events from text.
    
    Args:
        text: Raw text input
        client: LLM client instance
        
    Returns:
        List of EventDraft objects
    """
    import json
    from backend.llm.prompts import build_extract_event_prompt
    
    # Build extraction prompt
    prompt = build_extract_event_prompt(text)
    
    # Get LLM response
    response = client.generate(prompt, max_tokens=1024)
    
    # Parse JSON response
    try:
        events_data = json.loads(response.strip())
        if not isinstance(events_data, list):
            raise ValueError("Response is not a JSON array")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"⚠️  Failed to parse JSON response: {e}")
        print(f"⚠️  Raw response: {response[:200]}...")
        # Fallback to rule-based extraction
        extractor = TextExtractor()
        return [extractor.extract(text)]
    
    # Convert JSON to EventDraft objects
    drafts = []
    for event_data in events_data:
        try:
            draft = _json_to_event_draft(event_data, text)
            drafts.append(draft)
        except Exception as e:
            print(f"⚠️  Failed to convert event data: {e}, skipping event")
            continue
    
    # If no events extracted, fallback to rule-based
    if not drafts:
        print("⚠️  No events extracted by Qwen, using rule-based fallback")
        extractor = TextExtractor()
        drafts = [extractor.extract(text)]
    
    return drafts


def _json_to_event_draft(event_data: dict, original_text: str) -> EventDraft:
    """
    Convert JSON event data to EventDraft object.
    
    Args:
        event_data: JSON object with event fields
        original_text: Original input text
        
    Returns:
        EventDraft object
    """
    
    # Extract fields from JSON (handle None values)
    summary = event_data.get('summary') or ''
    time_text = event_data.get('time_text') or ''
    event_type = event_data.get('event_type') or ''
    emotion = event_data.get('emotion') or ''
    preferences = event_data.get('preferences') or []
    taboos = event_data.get('taboos') or []
    tags = event_data.get('tags') or []
    
    # Strip strings safely
    summary = summary.strip() if isinstance(summary, str) else str(summary)
    time_text = time_text.strip() if isinstance(time_text, str) else str(time_text)
    event_type = event_type.strip() if isinstance(event_type, str) else str(event_type)
    emotion = emotion.strip() if isinstance(emotion, str) else str(emotion)
    
    # Convert time_text to occurred_at (simple mapping)
    occurred_at = _parse_time_text(time_text) if time_text else None
    
    # Ensure lists are actually lists
    if isinstance(preferences, str):
        preferences = [preferences] if preferences else []
    if isinstance(taboos, str):
        taboos = [taboos] if taboos else []
    if isinstance(tags, str):
        tags = [tags] if tags else []
    
    # Validate required fields
    if not summary:
        summary = original_text[:100] + "..." if len(original_text) > 100 else original_text
    
    return EventDraft(
        summary=summary,
        raw_text=original_text,
        occurred_at=occurred_at,
        raw_time_text=time_text or None,
        event_type=event_type or None,
        emotion=emotion or None,
        preferences=preferences,
        taboos=taboos,
        tags=tags,
    )


def _parse_time_text(time_text: str) -> str:
    """
    Parse time text to ISO format string.
    
    Args:
        time_text: Natural language time description
        
    Returns:
        ISO format time string or None
    """
    from datetime import datetime, timedelta
    
    now = datetime.now()
    
    # Simple time parsing (can be enhanced)
    time_mappings = {
        '今天': now,
        '昨天': now - timedelta(days=1),
        '前天': now - timedelta(days=2),
        '明天': now + timedelta(days=1),
        '后天': now + timedelta(days=2),
        '上周': now - timedelta(weeks=1),
        '下周': now + timedelta(weeks=1),
        '上个月': now - timedelta(days=30),
        '下个月': now + timedelta(days=30),
    }
    
    for key, time_obj in time_mappings.items():
        if key in time_text:
            return time_obj.strftime("%Y-%m-%dT%H:%M:%S")
    
    # If no match, return current time
    return now.strftime("%Y-%m-%dT%H:%M:%S")


def ingest_manual(
    person_ids: List[int],
    raw_text: str,
    auto_index: bool = True
) -> Event:
    """
    Ingest manually entered text and create an Event.
    
    This function:
    1. Extracts event information from raw text (using rules, future: AI)
    2. Creates an Event record in the database
    3. Indexes the event in the vector store for RAG retrieval
    
    Args:
        person_ids: List of person IDs this event is associated with
        raw_text: Raw text describing the event
        auto_index: Whether to automatically index in vector store
        
    Returns:
        The created Event object
        
    Raises:
        ValueError: If person_ids is empty or raw_text is blank
    """
    # Validate input
    if not person_ids:
        raise ValueError("At least one person_id is required")
    
    if not raw_text or not raw_text.strip():
        raise ValueError("raw_text cannot be empty")
    
    # Extract event information
    drafts = extract_events(raw_text)
    
    if not drafts:
        raise ValueError("Could not extract any events from the text")
    
    # Use the first draft (future: handle multiple events)
    draft = drafts[0]
    
    # Convert to Event
    event = draft.to_event(person_ids)
    
    # Save to database
    event_repo = EventRepository()
    created_event = event_repo.create(event)
    
    # Index in vector store
    if auto_index and created_event.id:
        retriever = get_retriever()
        embedding_id = retriever.index_event(created_event)
        
        # Update event with embedding_id
        created_event.embedding_id = embedding_id
    
    return created_event


def ingest_batch(
    person_ids: List[int],
    texts: List[str],
    auto_index: bool = True
) -> List[Event]:
    """
    Ingest multiple texts at once.
    
    Args:
        person_ids: List of person IDs for all events
        texts: List of raw text inputs
        auto_index: Whether to automatically index in vector store
        
    Returns:
        List of created Event objects
    """
    events = []
    for text in texts:
        try:
            event = ingest_manual(person_ids, text, auto_index)
            events.append(event)
        except ValueError as e:
            print(f"Warning: Skipping text due to error: {e}")
    
    return events


def ingest_ocr(
    person_ids: List[int],
    image_path: str,
    auto_index: bool = True,
) -> Event:
    """
    Ingest an image (e.g., chat screenshot) via OCR and create an Event.

    This function:
    1. Runs OCR on the image to obtain text
    2. Reuses ingest_manual to extract event info and store it

    Args:
        person_ids: List of person IDs this event is associated with
        image_path: Path to the image file
        auto_index: Whether to automatically index in vector store

    Returns:
        The created Event object

    Raises:
        ValueError: If person_ids is empty or OCR result is empty
        RuntimeError: If OCR dependencies are not installed
    """
    if not person_ids:
        raise ValueError("At least one person_id is required")

    image_file = Path(image_path)
    if not image_file.exists():
        raise ValueError(f"Image file not found: {image_file}")

    try:
        from PIL import Image  # type: ignore[import]
        import pytesseract  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "OCR dependencies not installed. "
            "Please install pillow and pytesseract, and ensure Tesseract OCR "
            "is available on your system."
        ) from exc

    image = Image.open(str(image_file))
    text = pytesseract.image_to_string(image)

    if not text or not text.strip():
        raise ValueError(f"OCR produced empty text for image: {image_file}")

    return ingest_manual(person_ids, text, auto_index=auto_index)


def ingest_audio(
    person_ids: List[int],
    audio_path: str,
    auto_index: bool = True,
) -> Event:
    """
    Ingest an audio recording via speech-to-text and create an Event.

    This function:
    1. Uses a local speech-to-text model (e.g., Whisper) to transcribe audio
    2. Reuses ingest_manual to extract event info and store it

    The actual STT backend is optional and lazily imported to avoid heavy
    dependencies by default. Recommended backend: openai/whisper.

    Args:
        person_ids: List of person IDs this event is associated with
        audio_path: Path to the audio file
        auto_index: Whether to automatically index in vector store

    Returns:
        The created Event object

    Raises:
        ValueError: If person_ids is empty or transcription is empty
        RuntimeError: If STT dependencies are not installed
    """
    if not person_ids:
        raise ValueError("At least one person_id is required")

    audio_file = Path(audio_path)
    if not audio_file.exists():
        raise ValueError(f"Audio file not found: {audio_file}")

    # Lazy import to keep base dependencies light
    try:
        import whisper  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Audio ingestion requires the 'whisper' package and its "
            "dependencies (e.g., PyTorch, ffmpeg). "
            "Install it via: pip install -U openai-whisper "
            "and ensure ffmpeg is available on your system."
        ) from exc

    # Load model (can be configured via env in the future)
    model = whisper.load_model("base")
    result = model.transcribe(str(audio_file), language="zh")
    text = (result.get("text") or "").strip()

    if not text:
        raise ValueError(f"Transcription produced empty text for audio: {audio_file}")

    return ingest_manual(person_ids, text, auto_index=auto_index)
