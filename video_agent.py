"""
backend/video_agent.py  (YouTube RAG Agent)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Replaced: local video file upload + OpenCV frame extraction
Replaced with: YouTube URL → transcript + metadata → FAISS → semantic Q&A

How it works
────────────
  1. User pastes a YouTube URL  (any format: full, short, embed)
  2. fetch_youtube_data      — extracts video ID, fetches metadata (title,
                               channel, description, duration) via yt-dlp
                               or YouTube oEmbed as fallback
  3. fetch_transcript        — pulls timestamped transcript via
                               youtube-transcript-api (auto/manual captions)
  4. chunk_and_index         — splits transcript into overlapping chunks,
                               stores in FAISS with timestamp metadata
  5. query_youtube           — semantic retrieval + Gemini answer with
                               timestamp citations
  6. summarize_video         — full video summary from transcript

Agent pattern
─────────────
  Tools  →  create_agent  →  AgentExecutor

Dependencies (add to requirements.txt)
──────────────────────────────────────
  youtube-transcript-api>=0.6.2
  yt-dlp>=2024.1.0          (optional – used for rich metadata)
"""

from __future__ import annotations

import json
import re
import textwrap
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, parse_qs

from langchain.agents import create_agent, AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain.schema import Document
from langchain.tools import StructuredTool
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from .base import get_llm, build_vectorstore

# ── Optional dependencies ──────────────────────────────────────────────────────
try:
    from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
    TRANSCRIPT_API_AVAILABLE = True
except ImportError:
    TRANSCRIPT_API_AVAILABLE = False

try:
    import yt_dlp
    YTDLP_AVAILABLE = True
except ImportError:
    YTDLP_AVAILABLE = False

try:
    import requests as _requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SHARED STATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _YouTubeState:
    vectorstore  = None
    video_id:    str = ""
    video_url:   str = ""
    title:       str = ""
    channel:     str = ""
    description: str = ""
    duration:    str = ""
    thumbnail:   str = ""
    transcript_chunks: List[Dict] = []   # [{text, start, duration}]
    full_transcript:   str = ""

_state = _YouTubeState()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _extract_video_id(url: str) -> str:
    """
    Extract YouTube video ID from any URL format:
      • https://www.youtube.com/watch?v=VIDEO_ID
      • https://youtu.be/VIDEO_ID
      • https://www.youtube.com/embed/VIDEO_ID
      • https://youtube.com/shorts/VIDEO_ID
      • Raw 11-char video ID
    """
    url = url.strip()

    # Already a bare video ID (11 chars, alphanumeric + _ -)
    if re.match(r'^[A-Za-z0-9_\-]{11}$', url):
        return url

    patterns = [
        r'(?:v=|/v/|youtu\.be/|/embed/|/shorts/)([A-Za-z0-9_\-]{11})',
    ]
    for pat in patterns:
        m = re.search(pat, url)
        if m:
            return m.group(1)

    # Try urllib parse
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    if 'v' in qs:
        return qs['v'][0]

    raise ValueError(f"Cannot extract YouTube video ID from: {url!r}")


def _seconds_to_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS or MM:SS string."""
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TASK SCHEMAS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class FetchYouTubeInput(BaseModel):
    youtube_url: str = Field(
        description="YouTube video URL or video ID. "
                    "Accepts: full URL, youtu.be short link, embed link, or bare 11-char ID."
    )

class FetchTranscriptInput(BaseModel):
    language: str = Field(
        default="en",
        description="Preferred transcript language code (default 'en'). "
                    "Falls back to auto-generated captions if manual not found."
    )

class IndexTranscriptInput(BaseModel):
    chunk_size_seconds: int = Field(
        default=60,
        description="Group transcript lines into chunks of this many seconds (default 60)."
    )

class QueryYouTubeInput(BaseModel):
    question: str = Field(description="Question to answer from the YouTube video content.")

class SummarizeVideoInput(BaseModel):
    style: str = Field(
        default="detailed",
        description="Summary style: 'brief' (3-5 sentences) | 'detailed' (full sections) | 'bullets' (key points)"
    )

class GetVideoInfoInput(BaseModel):
    pass


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TASK 1 — FETCH YOUTUBE METADATA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _run_fetch_youtube(youtube_url: str) -> str:
    """Extract video ID and fetch title, channel, description via yt-dlp or oEmbed."""
    try:
        video_id = _extract_video_id(youtube_url)
    except ValueError as exc:
        return f"❌ {exc}"

    _state.video_id  = video_id
    _state.video_url = f"https://www.youtube.com/watch?v={video_id}"
    _state.thumbnail = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"

    # ── Try yt-dlp first (richest metadata) ──────────────────────────────────
    if YTDLP_AVAILABLE:
        try:
            ydl_opts = {
                "quiet": True,
                "no_warnings": True,
                "skip_download": True,
                "extract_flat": False,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(_state.video_url, download=False)
            _state.title       = info.get("title", "Unknown Title")
            _state.channel     = info.get("uploader", info.get("channel", "Unknown Channel"))
            _state.description = (info.get("description", "") or "")[:1000]
            dur = info.get("duration", 0)
            _state.duration = _seconds_to_timestamp(dur) if dur else "Unknown"
            return (
                f"✅ Fetched metadata via yt-dlp\n"
                f"📹 Title   : {_state.title}\n"
                f"📺 Channel : {_state.channel}\n"
                f"⏱ Duration: {_state.duration}\n"
                f"🔗 URL     : {_state.video_url}\n"
                f"🖼 Thumbnail: {_state.thumbnail}"
            )
        except Exception as e:
            pass  # fall through to oEmbed

    # ── Fallback: YouTube oEmbed API (no auth needed) ─────────────────────────
    if REQUESTS_AVAILABLE:
        try:
            oembed_url = f"https://www.youtube.com/oembed?url={_state.video_url}&format=json"
            resp = _requests.get(oembed_url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                _state.title   = data.get("title", "Unknown Title")
                _state.channel = data.get("author_name", "Unknown Channel")
                _state.description = ""
                _state.duration    = "Unknown"
                return (
                    f"✅ Fetched metadata via oEmbed\n"
                    f"📹 Title   : {_state.title}\n"
                    f"📺 Channel : {_state.channel}\n"
                    f"🔗 URL     : {_state.video_url}\n"
                    f"🖼 Thumbnail: {_state.thumbnail}"
                )
        except Exception:
            pass

    # ── Bare minimum: just set video_id ───────────────────────────────────────
    _state.title   = f"YouTube Video ({video_id})"
    _state.channel = "Unknown"
    _state.duration = "Unknown"
    return (
        f"✅ Video ID extracted: {video_id}\n"
        f"🔗 URL: {_state.video_url}\n"
        f"⚠️ Could not fetch metadata (install yt-dlp or requests for richer info)"
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TASK 2 — FETCH TRANSCRIPT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _run_fetch_transcript(language: str = "en") -> str:
    """Pull timestamped transcript using youtube-transcript-api."""
    if not _state.video_id:
        return "❌ Fetch video metadata first using fetch_youtube_data."

    if not TRANSCRIPT_API_AVAILABLE:
        return (
            "❌ youtube-transcript-api not installed.\n"
            "Run: pip install youtube-transcript-api"
        )

    try:
        # Try requested language first, then auto-generated, then any available
        transcript_list = YouTubeTranscriptApi.list_transcripts(_state.video_id)

        transcript = None
        # 1. Try manual transcript in requested language
        try:
            transcript = transcript_list.find_manually_created_transcript([language])
        except Exception:
            pass

        # 2. Try auto-generated in requested language
        if transcript is None:
            try:
                transcript = transcript_list.find_generated_transcript([language])
            except Exception:
                pass

        # 3. Any available transcript (translated if needed)
        if transcript is None:
            available = list(transcript_list)
            if available:
                transcript = available[0]

        if transcript is None:
            return f"❌ No transcript found for video {_state.video_id}."

        chunks = transcript.fetch()
        _state.transcript_chunks = chunks

        # Build full plain-text transcript with timestamps
        lines = [
            f"[{_seconds_to_timestamp(c['start'])}] {c['text']}"
            for c in chunks
        ]
        _state.full_transcript = "\n".join(lines)

        lang_used = getattr(transcript, 'language_code', language)
        return (
            f"✅ Transcript fetched ({len(chunks)} segments, lang={lang_used})\n"
            f"📝 Total words: ~{sum(len(c['text'].split()) for c in chunks):,}\n"
            f"First line: {lines[0] if lines else 'empty'}"
        )

    except TranscriptsDisabled:
        return "❌ Transcripts are disabled for this video."
    except NoTranscriptFound:
        return f"❌ No transcript found in language '{language}'. Try language='en' or another code."
    except Exception as exc:
        return f"❌ Transcript error: {exc}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TASK 3 — CHUNK & INDEX TRANSCRIPT INTO FAISS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _run_index_transcript(chunk_size_seconds: int = 60) -> str:
    """
    Group transcript segments into time-window chunks, embed into FAISS.
    Each Document stores the timestamp so answers can cite exact moments.
    """
    if not _state.transcript_chunks:
        return "❌ No transcript loaded. Run fetch_transcript first."

    documents: List[Document] = []
    current_text  = []
    current_start = None
    current_end   = 0.0

    for seg in _state.transcript_chunks:
        start = seg.get("start", 0)
        dur   = seg.get("duration", 0)
        text  = seg.get("text", "").strip()

        if current_start is None:
            current_start = start

        current_text.append(text)
        current_end = start + dur

        # Close chunk when we've covered chunk_size_seconds
        if (current_end - current_start) >= chunk_size_seconds:
            chunk_text = " ".join(current_text)
            ts_start   = _seconds_to_timestamp(current_start)
            ts_end     = _seconds_to_timestamp(current_end)
            documents.append(Document(
                page_content=f"[{ts_start} → {ts_end}] {chunk_text}",
                metadata={
                    "source":    _state.video_url,
                    "video_id":  _state.video_id,
                    "title":     _state.title,
                    "start_sec": current_start,
                    "end_sec":   current_end,
                    "timestamp": ts_start,
                }
            ))
            current_text  = []
            current_start = None

    # Flush remaining
    if current_text and current_start is not None:
        chunk_text = " ".join(current_text)
        ts_start   = _seconds_to_timestamp(current_start)
        ts_end     = _seconds_to_timestamp(current_end)
        documents.append(Document(
            page_content=f"[{ts_start} → {ts_end}] {chunk_text}",
            metadata={
                "source":    _state.video_url,
                "video_id":  _state.video_id,
                "title":     _state.title,
                "start_sec": current_start,
                "end_sec":   current_end,
                "timestamp": ts_start,
            }
        ))

    if not documents:
        return "❌ No chunks created — transcript may be empty."

    _state.vectorstore = build_vectorstore(documents)
    return (
        f"✅ Indexed {len(documents)} transcript chunks "
        f"({chunk_size_seconds}s windows) into FAISS.\n"
        f"📹 Video: {_state.title}"
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TASK 4 — QUERY VIDEO CONTENT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _run_query_youtube(question: str) -> str:
    """Semantic search over transcript chunks + Gemini answer with timestamp citations."""
    if _state.vectorstore is None:
        return json.dumps({
            "answer": "❌ Video not indexed yet. Run fetch_youtube_data → fetch_transcript → index_transcript first.",
            "timestamps": [],
            "video_url": "",
        })

    retriever = _state.vectorstore.as_retriever(search_kwargs={"k": 5})
    docs      = retriever.get_relevant_documents(question)

    if not docs:
        return json.dumps({
            "answer": "No relevant content found in the video transcript.",
            "timestamps": [],
            "video_url": _state.video_url,
        })

    context    = "\n\n".join(d.page_content for d in docs)
    timestamps = [
        {
            "timestamp": d.metadata.get("timestamp", ""),
            "start_sec": d.metadata.get("start_sec", 0),
            "yt_link":   f"{_state.video_url}&t={int(d.metadata.get('start_sec', 0))}s",
            "snippet":   d.page_content[:100] + "...",
        }
        for d in docs
    ]

    llm    = get_llm(temperature=0.1)
    prompt = (
        f"You are analyzing the YouTube video: \"{_state.title}\" by {_state.channel}.\n\n"
        "Use the transcript excerpts below to answer the question. "
        "Reference specific timestamps like [MM:SS] when citing content.\n\n"
        f"Transcript excerpts:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    resp = llm.invoke([HumanMessage(content=prompt)])

    return json.dumps({
        "answer":     resp.content,
        "timestamps": timestamps,
        "video_url":  _state.video_url,
        "title":      _state.title,
    })


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TASK 5 — SUMMARIZE VIDEO
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _run_summarize_video(style: str = "detailed") -> str:
    """Summarize the full video from its transcript."""
    if not _state.full_transcript:
        return "❌ No transcript loaded. Run fetch_youtube_data → fetch_transcript first."

    # Use first 12000 chars to stay within context limits
    transcript_excerpt = _state.full_transcript[:12000]
    if len(_state.full_transcript) > 12000:
        transcript_excerpt += "\n\n[Transcript truncated for length...]"

    llm = get_llm(temperature=0.2)

    if style == "brief":
        instruction = "Write a brief 3-5 sentence summary of this video."
    elif style == "bullets":
        instruction = (
            "Extract the 8-12 most important points from this video as bullet points. "
            "Each bullet should be specific and include a timestamp reference."
        )
    else:  # detailed
        instruction = (
            "Write a comprehensive summary with these sections:\n"
            "## Overview\n## Main Topics Covered\n## Key Insights & Takeaways\n"
            "## Notable Quotes or Moments\n## Conclusion\n\n"
            "Include timestamp references [MM:SS] where relevant."
        )

    prompt = (
        f"Video: \"{_state.title}\" by {_state.channel}\n"
        f"Duration: {_state.duration}\n\n"
        f"Transcript:\n{transcript_excerpt}\n\n"
        f"{instruction}"
    )

    resp = llm.invoke([HumanMessage(content=prompt)])
    return json.dumps({
        "summary":   resp.content,
        "title":     _state.title,
        "channel":   _state.channel,
        "duration":  _state.duration,
        "video_url": _state.video_url,
        "thumbnail": _state.thumbnail,
        "style":     style,
    })


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TASK 6 — GET VIDEO INFO
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _run_get_video_info() -> str:
    """Return current video metadata and index status."""
    if not _state.video_id:
        return json.dumps({"status": "No video loaded yet."})
    return json.dumps({
        "video_id":     _state.video_id,
        "title":        _state.title,
        "channel":      _state.channel,
        "duration":     _state.duration,
        "video_url":    _state.video_url,
        "thumbnail":    _state.thumbnail,
        "transcript_segments": len(_state.transcript_chunks),
        "indexed":      _state.vectorstore is not None,
        "description":  _state.description[:300] + "..." if len(_state.description) > 300 else _state.description,
    })


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TOOL DEFINITIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

fetch_youtube_data = StructuredTool.from_function(
    func=_run_fetch_youtube,
    name="fetch_youtube_data",
    description=(
        "Fetch metadata (title, channel, duration) for a YouTube video. "
        "Accepts any YouTube URL format or bare video ID. "
        "Always call this first before fetching the transcript."
    ),
    args_schema=FetchYouTubeInput,
)

fetch_transcript = StructuredTool.from_function(
    func=_run_fetch_transcript,
    name="fetch_transcript",
    description=(
        "Fetch the timestamped transcript/captions for the loaded YouTube video. "
        "Tries manual captions first, then auto-generated. "
        "Call fetch_youtube_data before this."
    ),
    args_schema=FetchTranscriptInput,
)

index_transcript = StructuredTool.from_function(
    func=_run_index_transcript,
    name="index_transcript",
    description=(
        "Chunk the transcript into time-window segments and index them in FAISS "
        "for semantic search. Call after fetch_transcript."
    ),
    args_schema=IndexTranscriptInput,
)

query_youtube = StructuredTool.from_function(
    func=_run_query_youtube,
    name="query_youtube",
    description=(
        "Answer a question about the YouTube video using semantic search over "
        "the indexed transcript. Returns answer with timestamp citations and "
        "clickable YouTube links to exact moments."
    ),
    args_schema=QueryYouTubeInput,
)

summarize_video = StructuredTool.from_function(
    func=_run_summarize_video,
    name="summarize_video",
    description=(
        "Generate a summary of the YouTube video (brief, detailed, or bullet points). "
        "Call after fetch_transcript."
    ),
    args_schema=SummarizeVideoInput,
)

get_video_info = StructuredTool.from_function(
    func=_run_get_video_info,
    name="get_video_info",
    description="Return current video metadata and index status.",
    args_schema=GetVideoInfoInput,
)

VIDEO_TOOLS = [
    fetch_youtube_data,
    fetch_transcript,
    index_transcript,
    query_youtube,
    summarize_video,
    get_video_info,
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AGENT FACTORY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def create_video_rag_agent() -> AgentExecutor:
    """
    Build a YouTube RAG agent using create_agent.

    Workflow the agent follows:
        1. fetch_youtube_data(youtube_url)
        2. fetch_transcript()
        3. index_transcript()
        4. query_youtube(question) / summarize_video()
    """
    llm   = get_llm(temperature=0.1)
    tools = VIDEO_TOOLS

    return create_agent(
        tools=tools, llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True, handle_parsing_errors=True, max_iterations=10,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONVENIENCE CLASS  (Streamlit-facing API)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class VideoRAGAgent:
    name        = "YouTube RAG Agent"
    description = (
        "Paste a YouTube URL to load the transcript, then ask any question about "
        "the video — get answers with clickable timestamp links."
    )

    def __init__(self):
        self._executor: Optional[AgentExecutor] = None

    @property
    def executor(self) -> AgentExecutor:
        if self._executor is None:
            self._executor = create_video_rag_agent()
        return self._executor

    # ── High-level convenience methods (used by Streamlit) ────────────────────

    def ingest(self, youtube_url: str, language: str = "en") -> str:
        """
        Full pipeline: fetch metadata → fetch transcript → index.
        Returns a status string.
        """
        results = []

        # Step 1 — metadata
        r1 = _run_fetch_youtube(youtube_url)
        results.append(r1)
        if r1.startswith("❌"):
            return r1

        # Step 2 — transcript
        r2 = _run_fetch_transcript(language)
        results.append(r2)
        if r2.startswith("❌"):
            return "\n".join(results)

        # Step 3 — index
        r3 = _run_index_transcript(chunk_size_seconds=60)
        results.append(r3)

        return "\n\n".join(results)

    def query(self, question: str) -> Dict[str, Any]:
        """Semantic Q&A with timestamp citations."""
        raw = _run_query_youtube(question)
        try:
            return json.loads(raw)
        except Exception:
            return {"answer": raw, "timestamps": [], "video_url": _state.video_url}

    def summarize(self, style: str = "detailed") -> Dict[str, Any]:
        """Get a video summary."""
        raw = _run_summarize_video(style)
        try:
            return json.loads(raw)
        except Exception:
            return {"summary": raw, "title": _state.title}

    def get_info(self) -> Dict[str, Any]:
        """Get current video metadata."""
        raw = _run_get_video_info()
        try:
            return json.loads(raw)
        except Exception:
            return {}

    @staticmethod
    def is_youtube_url(text: str) -> bool:
        """Quick check if a string looks like a YouTube URL."""
        return bool(re.search(r'(youtube\.com|youtu\.be)', text, re.IGNORECASE))
