"""
Stream Processor - A singleton utility for text transformation between streams.

Sits between stdin/stdout to format, encode/decode, prettify, and sanitize text.
Supports multiple formats and can be passed around anywhere in your application.

LIBRARY ECOSYSTEM NOTES:
========================
This module is designed to work standalone but can integrate with these libraries:

ENCODING DETECTION & FIXING:
- ftfy: Fixes mojibake (encoding mix-ups) - pip install ftfy
  * fix_text() repairs double-encoded UTF-8, curly quotes, HTML entities
  * fix_and_explain() shows what encoding errors occurred
  * Cannot be used for AI training per license
  
- charset-normalizer: Modern chardet replacement - pip install charset-normalizer
  * 2-5x faster than chardet, better accuracy
  * from_bytes() returns multiple encoding candidates with confidence
  * Used by requests library
  
- chardet: Legacy encoding detection - pip install chardet
  * detect() returns encoding guess with confidence
  * Older, less accurate on modern Unicode/emoji

HTML SANITIZATION:
- bleach: Allowlist-based HTML sanitizer - pip install bleach
  * clean() strips/escapes disallowed tags and attributes  
  * linkify() auto-links URLs safely
  * Built on html5lib for browser-accurate parsing
  
- html-sanitizer: Opinionated HTML cleaner - pip install html-sanitizer
  * More aggressive cleanup than bleach
  * Converts <b>/<i> to <strong>/<em>
  * Strips all inline styles

- BeautifulSoup: HTML/XML parser - pip install beautifulsoup4
  * prettify() formats HTML with indentation
  * Robust against malformed markup

TERMINAL FORMATTING:
- Rich: Beautiful terminal output - pip install rich
  * Console markup: [bold red]text[/bold red]
  * Tables, progress bars, syntax highlighting
  * export_html/svg/text for recording output
  
WINDOWS ENCODING ISSUES:
========================
Common problems this module addresses:

1. UTF-8 BOM (Byte Order Mark)
   - Microsoft uses 0xEF 0xBB 0xBF prefix for UTF-8 files
   - Use 'utf-8-sig' encoding to handle this automatically
   - Notepad, Excel exports often include BOM

2. CP1252 / Windows-1252 vs UTF-8
   - Windows default encoding is CP1252, not UTF-8
   - Characters 0x80-0x9F are different between CP1252 and Latin-1
   - "Smart quotes" (0x93, 0x94) often cause issues
   - Euro symbol (€) is 0x80 in CP1252

3. Mojibake Patterns
   - UTF-8 decoded as CP1252: â€" instead of —
   - UTF-8 decoded as Latin-1: Ã© instead of é
   - Double-encoding: Ã©Ã© from encoding twice

4. Excel CSV Exports  
   - Excel exports use system locale encoding (often CP1252)
   - "Unicode Text" export uses UTF-16 with tabs
   - Recommendation: Use Google Sheets for UTF-8 CSV

5. Line Endings
   - Windows: CRLF (\\r\\n)
   - Unix/Mac: LF (\\n)
   - Old Mac: CR (\\r)

PROJECT DIRECTION / FUTURE STUBS:
=================================
The following areas could be expanded:

1. Pluggable Backend System
   - Allow swapping in ftfy, bleach, Rich as backends
   - Factory pattern for sanitizers, prettifiers
   
2. Streaming Pipeline
   - Generator-based processing for large files
   - Chunked encoding detection
   
3. Format Converters
   - Markdown -> HTML -> Plain text
   - Rich terminal -> HTML for logs
   
4. Language Detection
   - charset-normalizer includes language detection
   - Could inform encoding decisions
   
5. Content-Type Negotiation  
   - HTTP Content-Type header parsing
   - MIME type to format mapping
"""

from __future__ import annotations

import codecs
import html
import io
import json
import re
import sys
import textwrap
import unicodedata
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import lru_cache
from html.parser import HTMLParser
from typing import (
    Any,
    Callable,
    Generator,
    IO,
    Iterator,
    Literal,
    Protocol,
    TextIO,
    TypeVar,
    runtime_checkable,
)

T = TypeVar("T")


# =============================================================================
# Enums and Constants
# =============================================================================

class TextFormat(Enum):
    """Supported text formats for processing."""
    PLAIN = auto()
    HTML = auto()
    MARKDOWN = auto()
    JSON = auto()
    XML = auto()
    ANSI = auto()
    CSV = auto()
    RTF = auto()
    YAML = auto()


class Encoding(Enum):
    """
    Common text encodings with Windows-specific variants.
    
    Windows Encoding Notes:
    - CP1252 (Windows-1252): Western European Windows default
    - CP1250: Central/Eastern European Windows
    - CP1251: Cyrillic Windows  
    - CP1253: Greek Windows
    - CP1254: Turkish Windows
    - CP1255: Hebrew Windows
    - CP1256: Arabic Windows
    - CP1257: Baltic Windows
    - CP1258: Vietnamese Windows
    """
    # Unicode variants
    UTF8 = "utf-8"
    UTF8_SIG = "utf-8-sig"  # UTF-8 with BOM (Windows Notepad)
    UTF16 = "utf-16"
    UTF16_LE = "utf-16-le"  # Little-endian (Windows default)
    UTF16_BE = "utf-16-be"  # Big-endian
    UTF32 = "utf-32"
    
    # ASCII and Latin
    ASCII = "ascii"
    LATIN1 = "latin-1"  # ISO-8859-1
    
    # Windows codepages
    CP1252 = "cp1252"   # Western European (most common Windows)
    CP1250 = "cp1250"   # Central European
    CP1251 = "cp1251"   # Cyrillic
    CP1253 = "cp1253"   # Greek
    CP1254 = "cp1254"   # Turkish
    CP1256 = "cp1256"   # Arabic
    
    # DOS codepages (legacy)
    CP437 = "cp437"     # Original IBM PC / DOS
    CP850 = "cp850"     # DOS Latin-1
    
    # Mac
    MAC_ROMAN = "mac-roman"


# BOM (Byte Order Mark) signatures for detection
BOM_SIGNATURES: dict[bytes, str] = {
    codecs.BOM_UTF8: "utf-8-sig",
    codecs.BOM_UTF16_LE: "utf-16-le", 
    codecs.BOM_UTF16_BE: "utf-16-be",
    codecs.BOM_UTF32_LE: "utf-32-le",
    codecs.BOM_UTF32_BE: "utf-32-be",
}

# Common mojibake patterns (UTF-8 decoded as wrong encoding)
# Format: mojibake_string -> correct_string
MOJIBAKE_PATTERNS: dict[str, str] = {
    # UTF-8 decoded as CP1252 / Latin-1 - common accented letters
    # These appear when UTF-8 bytes are interpreted as single-byte encoding
    "\xc3\xa9": "\xe9",      # é (e-acute)
    "\xc3\xa8": "\xe8",      # è (e-grave)
    "\xc3\xa0": "\xe0",      # à (a-grave)
    "\xc3\xa2": "\xe2",      # â (a-circumflex)
    "\xc3\xae": "\xee",      # î (i-circumflex)
    "\xc3\xaf": "\xef",      # ï (i-umlaut)
    "\xc3\xb4": "\xf4",      # ô (o-circumflex)
    "\xc3\xbb": "\xfb",      # û (u-circumflex)
    "\xc3\xa7": "\xe7",      # ç (c-cedilla)
    "\xc3\xb1": "\xf1",      # ñ (n-tilde)
    "\xc3\xbc": "\xfc",      # ü (u-umlaut)
    "\xc3\xb6": "\xf6",      # ö (o-umlaut)
    "\xc3\xa4": "\xe4",      # ä (a-umlaut)
    "\xc3\x89": "\xc9",      # É (E-acute)
    "\xc3\x80": "\xc0",      # À (A-grave)
    "\xc3\x9c": "\xdc",      # Ü (U-umlaut)
    "\xc3\x96": "\xd6",      # Ö (O-umlaut)
    "\xc3\x84": "\xc4",      # Ä (A-umlaut)
    "\xc3\x9f": "\xdf",      # ß (sharp s / eszett)
    # Common punctuation mojibake
    "\xe2\x80\x93": "\u2013",  # en dash
    "\xe2\x80\x94": "\u2014",  # em dash
    "\xe2\x80\x98": "\u2018",  # left single quote '
    "\xe2\x80\x99": "\u2019",  # right single quote '
    "\xe2\x80\x9c": "\u201c",  # left double quote "
    "\xe2\x80\x9d": "\u201d",  # right double quote "
    "\xe2\x80\xa6": "\u2026",  # ellipsis …
    "\xe2\x80\xa2": "\u2022",  # bullet •
    "\xe2\x82\xac": "\u20ac",  # Euro sign €
    "\xc2\xa0": "\xa0",        # non-breaking space
    "\xc2\xa9": "\xa9",        # copyright ©
    "\xc2\xae": "\xae",        # registered ®
    "\xc2\xb0": "\xb0",        # degree °
}


# =============================================================================
# Protocols for Plugin System
# =============================================================================

@runtime_checkable
class Sanitizer(Protocol):
    """Protocol for text sanitizers (bleach-compatible)."""
    def clean(self, text: str, **kwargs: Any) -> str: ...


@runtime_checkable  
class EncodingDetector(Protocol):
    """Protocol for encoding detectors (chardet-compatible)."""
    def detect(self, data: bytes) -> dict[str, Any]: ...


@runtime_checkable
class TextFixer(Protocol):
    """Protocol for text fixers (ftfy-compatible)."""
    def fix_text(self, text: str, **kwargs: Any) -> str: ...


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ProcessorConfig:
    """Configuration for the stream processor."""
    
    # Display settings
    line_width: int = 80
    indent_size: int = 4
    strip_trailing_whitespace: bool = True
    normalize_newlines: bool = True
    
    # Encoding settings
    default_encoding: Encoding = Encoding.UTF8
    fallback_encodings: tuple[Encoding, ...] = (
        Encoding.UTF8_SIG,   # Try UTF-8 with BOM first
        Encoding.CP1252,     # Common Windows encoding
        Encoding.LATIN1,     # Fallback that accepts any byte
    )
    encoding_errors: Literal["strict", "ignore", "replace", "backslashreplace"] = "replace"
    detect_bom: bool = True
    strip_bom: bool = True
    
    # Mojibake fixing
    fix_mojibake: bool = True
    mojibake_aggressive: bool = False  # Try harder fixes (may have false positives)
    
    # Sanitization settings
    strip_html_tags: bool = False
    escape_html_entities: bool = False
    remove_control_chars: bool = True
    normalize_unicode: bool = True
    unicode_form: Literal["NFC", "NFD", "NFKC", "NFKD"] = "NFC"
    
    # ANSI settings
    strip_ansi_codes: bool = False
    
    # Prettify settings
    auto_detect_format: bool = True
    json_indent: int = 2
    xml_indent: int = 2
    
    # Plugin backends (set to library instances if available)
    sanitizer_backend: Sanitizer | None = None  # e.g., bleach
    encoding_detector: EncodingDetector | None = None  # e.g., chardet
    text_fixer: TextFixer | None = None  # e.g., ftfy


# =============================================================================
# HTML Processing
# =============================================================================

class HTMLStripper(HTMLParser):
    """Parser that strips HTML tags and extracts text content."""
    
    def __init__(self) -> None:
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self._fed: list[str] = []
        self._in_script_or_style = False
    
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in ("script", "style"):
            self._in_script_or_style = True
    
    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in ("script", "style"):
            self._in_script_or_style = False
    
    def handle_data(self, data: str) -> None:
        if not self._in_script_or_style:
            self._fed.append(data)
    
    def get_text(self) -> str:
        return "".join(self._fed)
    
    @classmethod
    def strip(cls, html_text: str) -> str:
        """Strip HTML tags from text."""
        stripper = cls()
        try:
            stripper.feed(html_text)
        except Exception:
            # Fallback for malformed HTML
            return re.sub(r"<[^>]+>", "", html_text)
        return stripper.get_text()


# =============================================================================
# Transform Results
# =============================================================================

@dataclass
class TransformResult:
    """Result of a text transformation."""
    text: str
    original_format: TextFormat
    detected_encoding: str | None = None
    source_had_bom: bool = False
    mojibake_fixed: bool = False
    warnings: list[str] = field(default_factory=list)
    
    def __str__(self) -> str:
        return self.text


@dataclass  
class EncodingResult:
    """Result of encoding detection."""
    encoding: str
    confidence: float
    language: str | None = None
    had_bom: bool = False
    
    @classmethod
    def from_chardet(cls, result: dict[str, Any]) -> EncodingResult:
        """Create from chardet/charset-normalizer result."""
        return cls(
            encoding=result.get("encoding", "utf-8") or "utf-8",
            confidence=result.get("confidence", 0.0) or 0.0,
            language=result.get("language"),
        )


# =============================================================================
# Main Processor
# =============================================================================

class StreamProcessor:
    """
    Singleton stream processor for text transformation.
    
    Handles formatting, encoding/decoding, prettification, and sanitization
    of text streams. Can be used as a context manager to intercept stdout/stdin.
    
    Usage:
        processor = StreamProcessor.instance()
        
        # Direct processing
        result = processor.process("some text")
        
        # Stream interception
        with processor.intercept_stdout():
            print("This will be processed")
        
        # Chained transforms
        result = (processor
            .chain(text)
            .sanitize()
            .prettify()
            .wrap(width=80)
            .result())
        
        # With external libraries
        import ftfy
        import bleach
        config = ProcessorConfig(
            text_fixer=ftfy,
            sanitizer_backend=bleach,
        )
        processor = StreamProcessor.instance(config)
    """
    
    _instance: StreamProcessor | None = None
    _initialized: bool = False
    
    def __new__(cls, config: ProcessorConfig | None = None) -> StreamProcessor:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config: ProcessorConfig | None = None) -> None:
        if StreamProcessor._initialized:
            if config is not None:
                self._config = config
            return
        
        self._config = config or ProcessorConfig()
        self._original_stdout: TextIO | None = None
        self._original_stdin: TextIO | None = None
        
        # Compile regex patterns once
        self._ansi_pattern = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")
        self._ansi_osc_pattern = re.compile(r"\x1b\][^\x07]*\x07")  # OSC sequences
        self._control_pattern = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
        self._whitespace_pattern = re.compile(r"[ \t]+$", re.MULTILINE)
        self._multi_newline_pattern = re.compile(r"\n{3,}")
        self._c1_control_pattern = re.compile(r"[\x80-\x9f]")
        
        StreamProcessor._initialized = True
    
    @classmethod
    def instance(cls, config: ProcessorConfig | None = None) -> StreamProcessor:
        """Get the singleton instance."""
        return cls(config)
    
    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton (primarily for testing)."""
        cls._instance = None
        cls._initialized = False
    
    @property
    def config(self) -> ProcessorConfig:
        """Current configuration."""
        return self._config
    
    @config.setter
    def config(self, value: ProcessorConfig) -> None:
        """Update configuration."""
        self._config = value
    
    # =========================================================================
    # BOM Detection
    # =========================================================================
    
    def detect_bom(self, data: bytes) -> tuple[str | None, int]:
        """
        Detect BOM (Byte Order Mark) in data.
        
        Returns:
            Tuple of (encoding_name, bom_length) or (None, 0) if no BOM.
        
        Windows Note:
            Microsoft Notepad and many Windows apps prepend UTF-8 BOM (0xEF 0xBB 0xBF)
            to UTF-8 files. This is technically valid but causes issues when
            the file is read without BOM awareness.
        """
        for bom, encoding in BOM_SIGNATURES.items():
            if data.startswith(bom):
                return encoding, len(bom)
        return None, 0
    
    # =========================================================================
    # Encoding Detection
    # =========================================================================
    
    def detect_encoding(self, data: bytes) -> EncodingResult:
        """
        Detect the encoding of byte data.
        
        Uses external detector if configured, otherwise uses heuristics.
        """
        # Check for BOM first
        bom_encoding, bom_len = self.detect_bom(data)
        if bom_encoding:
            return EncodingResult(
                encoding=bom_encoding,
                confidence=1.0,
                had_bom=True,
            )
        
        # Use external detector if available
        if self._config.encoding_detector:
            result = self._config.encoding_detector.detect(data)
            return EncodingResult.from_chardet(result)
        
        # Built-in heuristics
        return self._detect_encoding_heuristic(data)
    
    def _detect_encoding_heuristic(self, data: bytes) -> EncodingResult:
        """Simple encoding detection heuristics."""
        # Try UTF-8 first (most common modern encoding)
        try:
            data.decode("utf-8", errors="strict")
            return EncodingResult(encoding="utf-8", confidence=0.9)
        except UnicodeDecodeError:
            pass
        
        # Check for high bytes that suggest specific encodings
        high_bytes = [b for b in data if b > 127]
        
        if not high_bytes:
            # Pure ASCII
            return EncodingResult(encoding="ascii", confidence=1.0)
        
        # Check for Windows-1252 specific characters (0x80-0x9F range)
        # These are undefined in ISO-8859-1 but defined in CP1252
        cp1252_specific = {0x80, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88,
                          0x89, 0x8A, 0x8B, 0x8C, 0x8E, 0x91, 0x92, 0x93,
                          0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0x9B,
                          0x9C, 0x9E, 0x9F}
        
        if any(b in cp1252_specific for b in high_bytes):
            return EncodingResult(encoding="cp1252", confidence=0.7)
        
        # Default to latin-1 (accepts any byte)
        return EncodingResult(encoding="latin-1", confidence=0.5)
    
    # =========================================================================
    # Encoding / Decoding  
    # =========================================================================
    
    def decode(
        self,
        data: bytes,
        encoding: Encoding | str | None = None,
        errors: str | None = None,
    ) -> str:
        """
        Decode bytes to string with intelligent fallback handling.
        
        Handles Windows-specific issues:
        - UTF-8 BOM detection and stripping
        - CP1252 fallback for Windows-generated content
        - Multiple encoding attempts with fallback chain
        """
        if not data:
            return ""
            
        errors = errors or self._config.encoding_errors
        
        # Detect and handle BOM
        bom_encoding, bom_len = self.detect_bom(data) if self._config.detect_bom else (None, 0)
        
        if bom_encoding:
            if self._config.strip_bom:
                data = data[bom_len:]
            encoding = bom_encoding
        
        # Determine encoding to try
        if encoding is None:
            encoding = self._config.default_encoding
        
        if isinstance(encoding, Encoding):
            encoding = encoding.value
        
        # Try primary encoding
        try:
            return data.decode(encoding, errors="strict")
        except (UnicodeDecodeError, LookupError):
            pass
        
        # Try fallback chain
        for fallback in self._config.fallback_encodings:
            fallback_str = fallback.value if isinstance(fallback, Encoding) else fallback
            try:
                return data.decode(fallback_str, errors="strict")
            except (UnicodeDecodeError, LookupError):
                continue
        
        # Last resort: latin-1 with error handling (accepts any byte)
        return data.decode("latin-1", errors=errors)
    
    def encode(
        self,
        text: str,
        encoding: Encoding | str | None = None,
        errors: str | None = None,
        include_bom: bool = False,
    ) -> bytes:
        """
        Encode string to bytes.
        
        Args:
            text: String to encode
            encoding: Target encoding
            errors: Error handling strategy
            include_bom: Whether to prepend BOM (useful for Windows compatibility)
        """
        if encoding is None:
            encoding = self._config.default_encoding
        
        if isinstance(encoding, Encoding):
            encoding = encoding.value
        
        errors = errors or self._config.encoding_errors
        result = text.encode(encoding, errors=errors)
        
        if include_bom:
            if encoding == "utf-8":
                result = codecs.BOM_UTF8 + result
            elif encoding in ("utf-16", "utf-16-le"):
                result = codecs.BOM_UTF16_LE + result
            elif encoding == "utf-16-be":
                result = codecs.BOM_UTF16_BE + result
        
        return result
    
    # =========================================================================
    # Mojibake Detection and Fixing
    # =========================================================================
    
    def fix_mojibake(self, text: str) -> tuple[str, bool]:
        """
        Attempt to fix mojibake (encoding mix-ups).
        
        Uses external fixer (ftfy) if available, otherwise uses built-in patterns.
        
        Returns:
            Tuple of (fixed_text, was_modified)
        """
        if not self._config.fix_mojibake:
            return text, False
        
        # Use ftfy if available
        if self._config.text_fixer:
            fixed = self._config.text_fixer.fix_text(text)
            return fixed, fixed != text
        
        # Built-in mojibake fixing
        return self._fix_mojibake_builtin(text)
    
    def _fix_mojibake_builtin(self, text: str) -> tuple[str, bool]:
        """Built-in mojibake detection and repair."""
        original = text
        
        # Fix known mojibake patterns
        for bad, good in MOJIBAKE_PATTERNS.items():
            text = text.replace(bad, good)
        
        # Fix C1 control characters (0x80-0x9F) that should be CP1252
        # This happens when CP1252 is decoded as Latin-1
        def fix_c1(match: re.Match[str]) -> str:
            char = match.group(0)
            byte_val = ord(char)
            # Try to decode as CP1252
            try:
                return bytes([byte_val]).decode("cp1252")
            except (UnicodeDecodeError, ValueError):
                return char
        
        if self._c1_control_pattern.search(text):
            text = self._c1_control_pattern.sub(fix_c1, text)
        
        # Aggressive mode: try re-encoding detection
        if self._config.mojibake_aggressive and text == original:
            # Try UTF-8 encoded as Latin-1
            try:
                test = text.encode("latin-1").decode("utf-8")
                if test != text and self._looks_better(test, text):
                    text = test
            except (UnicodeEncodeError, UnicodeDecodeError):
                pass
            
            # Try UTF-8 encoded as CP1252
            try:
                test = text.encode("cp1252").decode("utf-8")
                if test != text and self._looks_better(test, text):
                    text = test
            except (UnicodeEncodeError, UnicodeDecodeError):
                pass
        
        return text, text != original
    
    def _looks_better(self, candidate: str, original: str) -> bool:
        """Heuristic: does candidate look more like valid text than original?"""
        # Count "weird" characters (replacement chars, C1 controls, etc.)
        def weirdness(s: str) -> int:
            count = 0
            for c in s:
                cp = ord(c)
                if cp == 0xFFFD:  # Replacement character
                    count += 10
                elif 0x80 <= cp <= 0x9F:  # C1 controls
                    count += 5
                elif unicodedata.category(c) == "Cc":  # Other control
                    count += 3
            return count
        
        return weirdness(candidate) < weirdness(original)
    
    # =========================================================================
    # Format Detection
    # =========================================================================
    
    @lru_cache(maxsize=128)
    def detect_format(self, text: str) -> TextFormat:
        """Auto-detect the format of text content."""
        text_sample = text[:2000].strip()
        
        # Check for JSON
        if text_sample.startswith(("{", "[")):
            try:
                json.loads(text)
                return TextFormat.JSON
            except json.JSONDecodeError:
                pass
        
        # Check for XML/HTML
        if text_sample.startswith("<?xml") or text_sample.startswith("<!DOCTYPE"):
            return TextFormat.XML
        
        if re.search(r"<\s*(html|head|body|div|span|p|a|script|style)\b", text_sample, re.I):
            return TextFormat.HTML
        
        if re.search(r"<\s*\w+[^>]*>.*</\s*\w+\s*>", text_sample, re.DOTALL):
            return TextFormat.XML
        
        # Check for YAML
        if re.match(r"^---\s*$", text_sample, re.MULTILINE) or \
           re.match(r"^\w+:\s*\S", text_sample, re.MULTILINE):
            # Basic YAML detection
            lines = text_sample.split("\n")[:10]
            yaml_like = sum(1 for l in lines if re.match(r"^\s*[\w-]+:\s*", l))
            if yaml_like >= 2:
                return TextFormat.YAML
        
        # Check for ANSI codes
        if self._ansi_pattern.search(text_sample):
            return TextFormat.ANSI
        
        # Check for Markdown indicators
        md_patterns = [
            r"^#{1,6}\s+",           # Headers
            r"^\s*[-*+]\s+",         # Lists
            r"\[.+\]\(.+\)",         # Links
            r"^\s*```",              # Code blocks
            r"\*\*.+\*\*",           # Bold
            r"__.+__",               # Bold alt
            r"^\s*>\s+",             # Blockquotes
        ]
        md_score = sum(1 for p in md_patterns if re.search(p, text_sample, re.MULTILINE))
        if md_score >= 2:
            return TextFormat.MARKDOWN
        
        # Check for CSV
        lines = text_sample.split("\n")[:5]
        if len(lines) >= 2:
            comma_counts = [line.count(",") for line in lines if line.strip()]
            if comma_counts and all(c == comma_counts[0] and c > 0 for c in comma_counts):
                return TextFormat.CSV
        
        # Check for RTF
        if text_sample.startswith("{\\rtf"):
            return TextFormat.RTF
        
        return TextFormat.PLAIN
    
    # =========================================================================
    # Sanitization
    # =========================================================================
    
    def sanitize(
        self,
        text: str,
        strip_html: bool | None = None,
        escape_html: bool | None = None,
        remove_control: bool | None = None,
        normalize: bool | None = None,
    ) -> str:
        """Sanitize text by removing/escaping dangerous content."""
        cfg = self._config
        
        # Normalize unicode first if requested
        if normalize if normalize is not None else cfg.normalize_unicode:
            text = unicodedata.normalize(cfg.unicode_form, text)
        
        # Remove control characters (but keep common whitespace)
        if remove_control if remove_control is not None else cfg.remove_control_chars:
            text = self._control_pattern.sub("", text)
        
        # Strip HTML tags
        if strip_html if strip_html is not None else cfg.strip_html_tags:
            if cfg.sanitizer_backend:
                text = cfg.sanitizer_backend.clean(text, tags=set(), strip=True)
            else:
                text = HTMLStripper.strip(text)
        
        # Escape HTML entities
        if escape_html if escape_html is not None else cfg.escape_html_entities:
            text = html.escape(text, quote=True)
        
        return text
    
    def sanitize_for_web(self, text: str) -> str:
        """
        Sanitize text for safe web display.
        
        Removes potential XSS vectors and unsafe content.
        """
        # Normalize unicode
        text = unicodedata.normalize("NFC", text)
        
        # Remove null bytes and control chars
        text = self._control_pattern.sub("", text)
        text = text.replace("\x00", "")
        
        # Escape HTML
        text = html.escape(text, quote=True)
        
        # Remove potential script injections even in escaped form
        text = re.sub(r"javascript\s*:", "", text, flags=re.I)
        text = re.sub(r"vbscript\s*:", "", text, flags=re.I)
        text = re.sub(r"data\s*:", "", text, flags=re.I)
        text = re.sub(r"on\w+\s*=", "", text, flags=re.I)
        
        return text
    
    def strip_html(self, text: str) -> str:
        """Remove all HTML tags from text."""
        if self._config.sanitizer_backend:
            return self._config.sanitizer_backend.clean(text, tags=set(), strip=True)
        return HTMLStripper.strip(text)
    
    def strip_ansi(self, text: str) -> str:
        """Remove ANSI escape codes from text."""
        text = self._ansi_pattern.sub("", text)
        text = self._ansi_osc_pattern.sub("", text)  # Also strip OSC sequences
        return text
    
    # =========================================================================
    # Prettification
    # =========================================================================
    
    def prettify(
        self,
        text: str,
        format_hint: TextFormat | None = None,
    ) -> str:
        """Prettify text based on detected or specified format."""
        if format_hint is None and self._config.auto_detect_format:
            format_hint = self.detect_format(text)
        
        format_hint = format_hint or TextFormat.PLAIN
        
        match format_hint:
            case TextFormat.JSON:
                return self._prettify_json(text)
            case TextFormat.XML:
                return self._prettify_xml(text)
            case TextFormat.HTML:
                return self._prettify_html(text)
            case TextFormat.MARKDOWN:
                return self._prettify_markdown(text)
            case TextFormat.YAML:
                return self._prettify_yaml(text)
            case TextFormat.ANSI:
                if self._config.strip_ansi_codes:
                    return self.strip_ansi(text)
                return text
            case _:
                return self._prettify_plain(text)
    
    def _prettify_json(self, text: str) -> str:
        """Prettify JSON content."""
        try:
            data = json.loads(text)
            return json.dumps(
                data,
                indent=self._config.json_indent,
                ensure_ascii=False,
                sort_keys=False,
            )
        except json.JSONDecodeError:
            return text
    
    def _prettify_xml(self, text: str) -> str:
        """Prettify XML content."""
        try:
            import xml.dom.minidom as minidom
            dom = minidom.parseString(text.encode())
            pretty = dom.toprettyxml(indent=" " * self._config.xml_indent)
            # Remove extra declaration if not in original
            lines = pretty.split("\n")
            if lines[0].startswith("<?xml") and not text.strip().startswith("<?xml"):
                return "\n".join(lines[1:]).strip()
            return pretty.strip()
        except Exception:
            return text
    
    def _prettify_html(self, text: str) -> str:
        """Prettify HTML content."""
        # Simple indentation-based prettification
        result: list[str] = []
        indent_level = 0
        indent = " " * self._config.xml_indent
        
        # Inline tags that shouldn't cause line breaks
        inline_tags = {"a", "abbr", "b", "em", "i", "span", "strong", "sub", "sup", "code"}
        
        # Split on tags
        parts = re.split(r"(<[^>]+>)", text)
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            if part.startswith("</"):
                indent_level = max(0, indent_level - 1)
                result.append(f"{indent * indent_level}{part}")
            elif part.startswith("<") and not part.startswith("<!"):
                tag_match = re.match(r"<(\w+)", part)
                tag_name = tag_match.group(1).lower() if tag_match else ""
                
                result.append(f"{indent * indent_level}{part}")
                
                if not part.endswith("/>") and tag_name not in inline_tags:
                    if not re.match(r"<(br|hr|img|input|meta|link|area|base|col|embed|param|source|track|wbr)", part, re.I):
                        indent_level += 1
            else:
                if part:
                    result.append(f"{indent * indent_level}{part}")
        
        return "\n".join(result)
    
    def _prettify_markdown(self, text: str) -> str:
        """Prettify Markdown content."""
        lines = text.split("\n")
        result: list[str] = []
        
        for line in lines:
            # Ensure headers have space after #
            line = re.sub(r"^(#{1,6})([^\s#])", r"\1 \2", line)
            # Normalize list markers
            line = re.sub(r"^(\s*)[*+-](\s+)", r"\1- \2", line)
            result.append(line)
        
        return "\n".join(result)
    
    def _prettify_yaml(self, text: str) -> str:
        """Prettify YAML content (basic)."""
        # Just normalize indentation
        lines = text.split("\n")
        return "\n".join(line.rstrip() for line in lines)
    
    def _prettify_plain(self, text: str) -> str:
        """Prettify plain text."""
        if self._config.strip_trailing_whitespace:
            text = self._whitespace_pattern.sub("", text)
        
        if self._config.normalize_newlines:
            text = text.replace("\r\n", "\n").replace("\r", "\n")
            text = self._multi_newline_pattern.sub("\n\n", text)
        
        return text
    
    # =========================================================================
    # Text Wrapping & Formatting
    # =========================================================================
    
    def wrap(
        self,
        text: str,
        width: int | None = None,
        indent: str = "",
        subsequent_indent: str | None = None,
    ) -> str:
        """Wrap text to specified width."""
        width = width or self._config.line_width
        subsequent_indent = subsequent_indent if subsequent_indent is not None else indent
        
        paragraphs = text.split("\n\n")
        wrapped_paragraphs: list[str] = []
        
        for para in paragraphs:
            if not para.strip():
                wrapped_paragraphs.append("")
                continue
            
            # Preserve existing line breaks for code/preformatted
            if para.startswith("    ") or para.startswith("\t"):
                wrapped_paragraphs.append(para)
                continue
            
            wrapped = textwrap.fill(
                para,
                width=width,
                initial_indent=indent,
                subsequent_indent=subsequent_indent,
                break_long_words=False,
                break_on_hyphens=True,
            )
            wrapped_paragraphs.append(wrapped)
        
        return "\n\n".join(wrapped_paragraphs)
    
    def indent(self, text: str, prefix: str | None = None) -> str:
        """Add indentation to text."""
        prefix = prefix or (" " * self._config.indent_size)
        return textwrap.indent(text, prefix)
    
    def dedent(self, text: str) -> str:
        """Remove common leading whitespace."""
        return textwrap.dedent(text)
    
    # =========================================================================
    # Stream Interception
    # =========================================================================
    
    @contextmanager
    def intercept_stdout(
        self,
        transform: Callable[[str], str] | None = None,
    ) -> Generator[io.StringIO, None, None]:
        """
        Context manager to intercept and process stdout.
        
        Usage:
            with processor.intercept_stdout() as captured:
                print("Hello")
            processed_output = captured.getvalue()
        """
        transform = transform or self.process
        
        class ProcessingStream(io.StringIO):
            def __init__(inner_self, processor_transform: Callable[[str], str]) -> None:
                super().__init__()
                inner_self._transform = processor_transform
                inner_self._original = sys.stdout
            
            def write(inner_self, s: str) -> int:
                processed = inner_self._transform(s)
                return super().write(processed)
        
        stream = ProcessingStream(transform)
        self._original_stdout = sys.stdout
        sys.stdout = stream  # type: ignore
        
        try:
            yield stream
        finally:
            sys.stdout = self._original_stdout
            self._original_stdout = None
    
    @contextmanager
    def intercept_stdin(
        self,
        transform: Callable[[str], str] | None = None,
        input_data: str | None = None,
    ) -> Generator[io.StringIO, None, None]:
        """
        Context manager to intercept and process stdin.
        
        Usage:
            with processor.intercept_stdin(input_data="hello\\n"):
                data = input()  # Returns processed "hello"
        """
        transform = transform or self.process
        
        class ProcessingInputStream(io.StringIO):
            def __init__(inner_self, data: str, processor_transform: Callable[[str], str]) -> None:
                processed = processor_transform(data)
                super().__init__(processed)
        
        input_data = input_data or ""
        stream = ProcessingInputStream(input_data, transform)
        self._original_stdin = sys.stdin
        sys.stdin = stream  # type: ignore
        
        try:
            yield stream
        finally:
            sys.stdin = self._original_stdin
            self._original_stdin = None
    
    @contextmanager
    def pipe(
        self,
        stdin_data: str | None = None,
    ) -> Generator[tuple[io.StringIO, io.StringIO], None, None]:
        """
        Intercept both stdin and stdout simultaneously.
        """
        with self.intercept_stdin(input_data=stdin_data) as stdin_stream:
            with self.intercept_stdout() as stdout_stream:
                yield stdin_stream, stdout_stream
    
    # =========================================================================
    # Main Processing
    # =========================================================================
    
    def process(
        self,
        text: str,
        format_hint: TextFormat | None = None,
        sanitize: bool = True,
        prettify: bool = True,
        fix_encoding: bool = True,
    ) -> str:
        """
        Main processing pipeline: fix encoding, sanitize, prettify, and format text.
        """
        if fix_encoding:
            text, _ = self.fix_mojibake(text)
        
        if sanitize:
            text = self.sanitize(text)
        
        if prettify:
            text = self.prettify(text, format_hint)
        
        return text
    
    def process_bytes(
        self,
        data: bytes,
        encoding: Encoding | str | None = None,
        **kwargs: Any,
    ) -> TransformResult:
        """Process raw bytes through the full pipeline."""
        # Detect encoding
        enc_result = self.detect_encoding(data)
        
        # Decode
        text = self.decode(data, encoding or enc_result.encoding)
        
        # Fix mojibake
        text, mojibake_fixed = self.fix_mojibake(text)
        
        # Process
        text = self.process(text, fix_encoding=False, **kwargs)
        
        # Detect format
        format_type = self.detect_format(text)
        
        return TransformResult(
            text=text,
            original_format=format_type,
            detected_encoding=enc_result.encoding,
            source_had_bom=enc_result.had_bom,
            mojibake_fixed=mojibake_fixed,
        )
    
    # =========================================================================
    # Fluent Interface / Transform Chain
    # =========================================================================
    
    def chain(self, text: str) -> TransformChain:
        """Start a fluent transform chain."""
        return TransformChain(self, text)
    
    # =========================================================================
    # Iterator Interface
    # =========================================================================
    
    def iter_lines(
        self,
        text: str,
        process_each: bool = True,
    ) -> Iterator[str]:
        """Iterate over processed lines."""
        for line in text.split("\n"):
            if process_each:
                yield self.process(line)
            else:
                yield line
    
    def iter_stream(
        self,
        stream: IO[str],
        process_each: bool = True,
    ) -> Iterator[str]:
        """Iterate over lines from a stream."""
        for line in stream:
            if process_each:
                yield self.process(line.rstrip("\n"))
            else:
                yield line.rstrip("\n")


# =============================================================================
# Fluent Transform Chain
# =============================================================================

class TransformChain:
    """
    Fluent interface for chaining text transformations.
    
    Usage:
        result = (StreamProcessor.instance()
            .chain(text)
            .decode()
            .fix_mojibake()
            .sanitize()
            .prettify()
            .wrap(width=80)
            .result())
    """
    
    def __init__(self, processor: StreamProcessor, text: str | bytes) -> None:
        self._processor = processor
        self._text = text if isinstance(text, str) else ""
        self._bytes = text if isinstance(text, bytes) else None
        self._format: TextFormat | None = None
        self._warnings: list[str] = []
        self._detected_encoding: str | None = None
        self._mojibake_fixed: bool = False
    
    def decode(
        self,
        encoding: Encoding | str | None = None,
    ) -> TransformChain:
        """Decode bytes to string."""
        if self._bytes is not None:
            enc_result = self._processor.detect_encoding(self._bytes)
            self._detected_encoding = enc_result.encoding
            self._text = self._processor.decode(self._bytes, encoding or enc_result.encoding)
            self._bytes = None
        return self
    
    def fix_mojibake(self) -> TransformChain:
        """Fix encoding mix-ups."""
        self._text, self._mojibake_fixed = self._processor.fix_mojibake(self._text)
        return self
    
    def sanitize(self, **kwargs: Any) -> TransformChain:
        """Apply sanitization."""
        self._text = self._processor.sanitize(self._text, **kwargs)
        return self
    
    def sanitize_for_web(self) -> TransformChain:
        """Apply web-safe sanitization."""
        self._text = self._processor.sanitize_for_web(self._text)
        return self
    
    def prettify(self, format_hint: TextFormat | None = None) -> TransformChain:
        """Apply prettification."""
        self._text = self._processor.prettify(self._text, format_hint or self._format)
        return self
    
    def wrap(self, width: int | None = None, **kwargs: Any) -> TransformChain:
        """Wrap text."""
        self._text = self._processor.wrap(self._text, width, **kwargs)
        return self
    
    def indent(self, prefix: str | None = None) -> TransformChain:
        """Indent text."""
        self._text = self._processor.indent(self._text, prefix)
        return self
    
    def dedent(self) -> TransformChain:
        """Dedent text."""
        self._text = self._processor.dedent(self._text)
        return self
    
    def strip_html(self) -> TransformChain:
        """Strip HTML tags."""
        self._text = self._processor.strip_html(self._text)
        return self
    
    def strip_ansi(self) -> TransformChain:
        """Strip ANSI codes."""
        self._text = self._processor.strip_ansi(self._text)
        return self
    
    def apply(self, func: Callable[[str], str]) -> TransformChain:
        """Apply a custom transformation."""
        self._text = func(self._text)
        return self
    
    def with_format(self, format_type: TextFormat) -> TransformChain:
        """Set format hint for subsequent operations."""
        self._format = format_type
        return self
    
    def normalize(self, form: Literal["NFC", "NFD", "NFKC", "NFKD"] = "NFC") -> TransformChain:
        """Normalize Unicode."""
        self._text = unicodedata.normalize(form, self._text)
        return self
    
    def replace(self, old: str, new: str) -> TransformChain:
        """Simple string replacement."""
        self._text = self._text.replace(old, new)
        return self
    
    def regex_replace(self, pattern: str, replacement: str, flags: int = 0) -> TransformChain:
        """Regex replacement."""
        self._text = re.sub(pattern, replacement, self._text, flags=flags)
        return self
    
    def result(self) -> TransformResult:
        """Get the final result."""
        return TransformResult(
            text=self._text,
            original_format=self._format or self._processor.detect_format(self._text),
            detected_encoding=self._detected_encoding,
            mojibake_fixed=self._mojibake_fixed,
            warnings=self._warnings,
        )
    
    def text(self) -> str:
        """Get just the text result."""
        return self._text
    
    def __str__(self) -> str:
        return self._text


# =============================================================================
# Convenience Functions (Module-level API)
# =============================================================================

def get_processor(config: ProcessorConfig | None = None) -> StreamProcessor:
    """Get the singleton processor instance."""
    return StreamProcessor.instance(config)


def process(text: str, **kwargs: Any) -> str:
    """Quick process using default singleton."""
    return get_processor().process(text, **kwargs)


def sanitize(text: str, **kwargs: Any) -> str:
    """Quick sanitize using default singleton."""
    return get_processor().sanitize(text, **kwargs)


def prettify(text: str, format_hint: TextFormat | None = None) -> str:
    """Quick prettify using default singleton."""
    return get_processor().prettify(text, format_hint)


def decode(data: bytes, encoding: str | None = None) -> str:
    """Quick decode using default singleton."""
    return get_processor().decode(data, encoding)


def fix_mojibake(text: str) -> str:
    """Quick mojibake fix using default singleton."""
    fixed, _ = get_processor().fix_mojibake(text)
    return fixed


# =============================================================================
# Integration Helpers
# =============================================================================

def configure_with_ftfy() -> ProcessorConfig:
    """
    Create config with ftfy integration.
    
    Usage:
        import ftfy
        config = configure_with_ftfy()
        config.text_fixer = ftfy
        processor = StreamProcessor.instance(config)
    """
    return ProcessorConfig(
        fix_mojibake=True,
        mojibake_aggressive=False,  # ftfy handles this
    )


def configure_with_bleach(tags: set[str] | None = None) -> ProcessorConfig:
    """
    Create config for use with bleach.
    
    Usage:
        import bleach
        config = configure_with_bleach()
        # config.sanitizer_backend = bleach  # bleach module itself works
        processor = StreamProcessor.instance(config)
    """
    return ProcessorConfig(
        strip_html_tags=True,
        escape_html_entities=False,  # bleach handles this
    )


def configure_for_windows() -> ProcessorConfig:
    """
    Create config optimized for Windows-generated content.
    
    Handles common Windows encoding issues:
    - UTF-8 BOM
    - CP1252 fallback
    - Smart quote normalization
    """
    return ProcessorConfig(
        default_encoding=Encoding.UTF8_SIG,
        fallback_encodings=(
            Encoding.UTF8,
            Encoding.CP1252,
            Encoding.LATIN1,
        ),
        detect_bom=True,
        strip_bom=True,
        fix_mojibake=True,
        normalize_newlines=True,  # Convert CRLF to LF
    )


# =============================================================================
# Demo / Testing
# =============================================================================

if __name__ == "__main__":
    # Demo usage
    processor = StreamProcessor.instance()
    
    print("=" * 60)
    print("STREAM PROCESSOR DEMO")
    print("=" * 60)
    
    # Test JSON prettification
    json_text = '{"name":"test","values":[1,2,3],"nested":{"key":"value"}}'
    print("\n=== JSON Prettification ===")
    print(processor.prettify(json_text))
    
    # Test HTML sanitization
    html_text = '<script>alert("xss")</script><p>Hello <b>World</b></p>'
    print("\n=== HTML Sanitization ===")
    print(f"Original: {html_text}")
    print(f"Stripped: {processor.strip_html(html_text)}")
    print(f"Web-safe: {processor.sanitize_for_web(html_text)}")
    
    # Test mojibake fixing
    print("\n=== Mojibake Fixing ===")
    mojibake_samples = [
        "Caf\xc3\xa9",      # UTF-8 e-acute decoded as Latin-1
        "\xe2\x80\x94test", # Em dash mojibake
        "na\xc3\xafve",     # UTF-8 i-umlaut decoded wrong
    ]
    for sample in mojibake_samples:
        fixed, was_fixed = processor.fix_mojibake(sample)
        status = "fixed" if was_fixed else "unchanged"
        print(f"  {sample!r:25} -> {fixed!r:15} ({status})")
    
    # Test Windows encoding handling
    print("\n=== Windows Encoding ===")
    # Simulate UTF-8 with BOM
    utf8_bom_bytes = codecs.BOM_UTF8 + "Hello World".encode("utf-8")
    result = processor.process_bytes(utf8_bom_bytes)
    print(f"UTF-8 BOM: detected={result.detected_encoding}, had_bom={result.source_had_bom}")
    print(f"Text: {result.text!r}")
    
    # Test fluent chain
    print("\n=== Fluent Chain ===")
    messy_text = '  {"data": "test"}  '
    result = (processor
        .chain(messy_text)
        .sanitize()
        .prettify()
        .result())
    print(f"Format detected: {result.original_format}")
    print(result.text)
    
    # Test stream interception
    print("\n=== Stream Interception ===")
    with processor.intercept_stdout() as captured:
        print('{"intercepted": true}')
    print(f"Captured and processed:\n{captured.getvalue()}")
    
    # Show encoding info
    print("\n=== Supported Encodings ===")
    print("Unicode: UTF-8, UTF-8-SIG (BOM), UTF-16, UTF-16-LE/BE, UTF-32")
    print("Windows: CP1252, CP1250, CP1251, CP1253, CP1254, CP1256")
    print("Legacy: ASCII, Latin-1, CP437, CP850, Mac-Roman")
    
    print("\n=== Configuration Examples ===")
    print("Windows content: configure_for_windows()")
    print("With ftfy:       configure_with_ftfy()")
    print("With bleach:     configure_with_bleach()")
