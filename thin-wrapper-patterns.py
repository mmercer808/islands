# Thin Wrapper Patterns

## 1. Simple Delegation Pattern
class APIWrapper:
    """Delegates calls to underlying implementation"""
    def __init__(self, implementation):
        self._impl = implementation
    
    def process(self, text):
        # Thin layer - just delegates
        return self._impl.process(text)

## 2. Proxy/Facade Pattern
class StreamFacade:
    """Simplifies complex subsystem"""
    def __init__(self):
        self._encoder = TextEncoder()
        self._compressor = Compressor()
        self._transmitter = Transmitter()
    
    def send(self, text):
        # Single simple interface to complex operations
        data = self._encoder.encode(text)
        compressed = self._compressor.compress(data)
        return self._transmitter.send(compressed)

## 3. Protocol/Interface Wrapper
from typing import Protocol

class StreamProtocol(Protocol):
    """Define interface without implementation"""
    def read(self) -> str: ...
    def write(self, data: str) -> None: ...

class FileStreamWrapper:
    """Wraps file to match protocol"""
    def __init__(self, file):
        self._file = file
    
    def read(self) -> str:
        return self._file.read()
    
    def write(self, data: str) -> None:
        self._file.write(data)

## 4. Lazy Proxy Pattern
class LazyAPIWrapper:
    """Delays expensive initialization"""
    def __init__(self, api_class, *args, **kwargs):
        self._api_class = api_class
        self._args = args
        self._kwargs = kwargs
        self._instance = None
    
    def _get_instance(self):
        if self._instance is None:
            self._instance = self._api_class(*self._args, **self._kwargs)
        return self._instance
    
    def __getattr__(self, name):
        # Only create instance when actually used
        return getattr(self._get_instance(), name)

## 5. Decorator Wrapper
class LoggingWrapper:
    """Adds functionality without changing interface"""
    def __init__(self, wrapped):
        self._wrapped = wrapped
    
    def __getattr__(self, name):
        attr = getattr(self._wrapped, name)
        if callable(attr):
            def wrapper(*args, **kwargs):
                print(f"Calling {name}")
                result = attr(*args, **kwargs)
                print(f"Completed {name}")
                return result
            return wrapper
        return attr

## 6. Adapter Pattern
class LegacyAPI:
    def fetch_text(self, id): 
        return f"Text {id}"

class ModernAPIAdapter:
    """Adapts old interface to new"""
    def __init__(self, legacy_api):
        self._legacy = legacy_api
    
    async def get_stream(self, id):
        # Adapt synchronous to async
        text = self._legacy.fetch_text(id)
        for char in text:
            yield char

## 7. Method Injection Wrapper
class ConfigurableWrapper:
    """Allows runtime behavior modification"""
    def __init__(self, processor=None):
        self._processor = processor or (lambda x: x)
    
    def set_processor(self, func):
        self._processor = func
    
    def process(self, text):
        return self._processor(text)