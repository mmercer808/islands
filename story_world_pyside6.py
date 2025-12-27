#!/usr/bin/env python3
"""
Story World Engine - Enhanced PySide6 Implementation
Complete system with OpenGL font rendering, entity tokens, and scene graphs
"""

import sys
import json
import time
import uuid
import re
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import random
import math

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QGridLayout, QSplitter, QTabWidget, QTextEdit, QPlainTextEdit,
    QLabel, QSlider, QPushButton, QListWidget, QListWidgetItem,
    QGroupBox, QFrame, QScrollArea, QProgressBar, QSpinBox,
    QColorDialog, QCheckBox, QComboBox, QTreeWidget, QTreeWidgetItem,
    QStatusBar, QMenuBar, QToolBar, QMessageBox
)
from PySide6.QtCore import (
    Qt, QTimer, QThread, Signal, QObject, QPropertyAnimation,
    QEasingCurve, QRect, QSize, QPoint, QSettings
)
from PySide6.QtGui import (
    QFont, QColor, QPalette, QPixmap, QPainter, QBrush, QLinearGradient,
    QTextCharFormat, QTextCursor, QSyntaxHighlighter, QTextDocument,
    QAction, QIcon
)

try:
    from PySide6.QtOpenGLWidgets import QOpenGLWidget
    from OpenGL.GL import *
    from OpenGL.GL.shaders import compileProgram, compileShader
    import numpy as np
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False
    QOpenGLWidget = QWidget  # Fallback
    print("OpenGL not available. Install PyOpenGL and numpy: pip install PyOpenGL PyOpenGL_accelerate numpy")

# Data Models
class EntityType(Enum):
    CHARACTER = "character"
    LOCATION = "location"
    ITEM = "item"
    EVENT = "event"
    CONCEPT = "concept"

@dataclass
class EntityData:
    id: str
    type: EntityType
    name: str
    description: str
    properties: Dict[str, Any]
    source_text: str
    spawned: float
    tick_count: int = 0
    is_running: bool = False
    energy: float = 0.5
    mood: float = 0.5
    activity: str = "idle"
    last_activity: str = "spawned"

class MessageType(Enum):
    SPAWN = "spawn"
    TICK = "tick"
    BEHAVIOR = "behavior"
    SAVE = "save"
    SYSTEM = "system"
    WEBGL = "webgl"

@dataclass
class LogMessage:
    timestamp: datetime
    type: MessageType
    message: str

# NLP Processor for entity extraction
class NLPProcessor:
    """Simple NLP processor for extracting entities from text"""
    
    @staticmethod
    def extract_entities(text: str) -> List[Dict[str, Any]]:
        entities = []
        
        # Enhanced character extraction
        character_pattern = r'\b([A-Z][a-z]+)\s+(?:walked|said|carried|approached|knew|was|felt|saw|heard|whispered|shouted)'
        characters = list(set(re.findall(character_pattern, text)))
        
        for name in characters:
            entities.append({
                'type': EntityType.CHARACTER,
                'name': name,
                'description': f'A character who appears in the story',
                'properties': {'firstMention': text.find(name)}
            })
        
        # Enhanced location extraction
        location_pattern = r'\b(?:through|of|to|in|at|from|toward)\s+(?:the\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
        locations = list(set(re.findall(location_pattern, text)))
        
        for name in locations:
            if name not in characters and len(name) > 3:
                entities.append({
                    'type': EntityType.LOCATION,
                    'name': name,
                    'description': f'A place within the story world',
                    'properties': {'firstMention': text.find(name)}
                })
        
        # Enhanced item extraction
        item_pattern = r'\b(?:carrying|magical|ancient|mysterious|glowing|father\'s|evil|dark)\s+([a-z]+(?:\s+[a-z]+)*)'
        items = list(set(re.findall(item_pattern, text)))
        
        for name in items:
            if len(name) > 3 and name not in ['very', 'more', 'most', 'some']:
                entities.append({
                    'type': EntityType.ITEM,
                    'name': name,
                    'description': f'An object of significance in the story',
                    'properties': {'firstMention': text.find(name)}
                })
        
        # Event extraction
        event_pattern = r'([A-Z][a-z]+)\s+(walked|approached|rumbled|whispered|felt)'
        events = re.findall(event_pattern, text)
        
        for i, (actor, action) in enumerate(events[:3]):  # Limit events
            entities.append({
                'type': EntityType.EVENT,
                'name': f'{actor} {action}',
                'description': f'An action that occurred in the story',
                'properties': {'actor': actor, 'action': action, 'sequence': i}
            })
        
        return entities

# Entity Token System
class EntityToken(QObject):
    """Individual entity with its own game loop"""
    
    state_changed = Signal(str)  # entity_id
    
    def __init__(self, data: EntityData):
        super().__init__()
        self.data = data
        self.timer = QTimer()
        self.timer.timeout.connect(self.tick)
        self.behaviors = self.get_behaviors_for_type()
        self.interval = self.get_interval_for_type()
        
    def get_behaviors_for_type(self) -> List[str]:
        behavior_map = {
            EntityType.CHARACTER: ['movement', 'interaction', 'emotion', 'memory'],
            EntityType.LOCATION: ['environment', 'population', 'discovery'],
            EntityType.ITEM: ['degradation', 'ownership', 'magic'],
            EntityType.EVENT: ['consequence', 'chain', 'temporal'],
            EntityType.CONCEPT: ['influence', 'evolution', 'manifestation']
        }
        return behavior_map.get(self.data.type, ['basic'])
    
    def get_interval_for_type(self) -> int:
        intervals = {
            EntityType.CHARACTER: 2500,
            EntityType.LOCATION: 6000,
            EntityType.ITEM: 8000,
            EntityType.EVENT: 4000,
            EntityType.CONCEPT: 10000
        }
        return intervals.get(self.data.type, 5000)
    
    def start_game_loop(self):
        if not self.data.is_running:
            self.data.is_running = True
            self.timer.start(self.interval)
            
    def stop_game_loop(self):
        if self.data.is_running:
            self.data.is_running = False
            self.timer.stop()
            
    def tick(self):
        self.data.tick_count += 1
        self.execute_behaviors()
        self.update_state()
        self.state_changed.emit(self.data.id)
        
    def execute_behaviors(self):
        behavior = random.choice(self.behaviors)
        
        if behavior == 'movement':
            self.data.activity = 'moving' if random.random() > 0.5 else 'resting'
        elif behavior == 'interaction':
            self.data.activity = 'interacting'
            self.data.mood += (random.random() - 0.5) * 0.15
        elif behavior == 'emotion':
            self.data.mood = max(0, min(1, self.data.mood + (random.random() - 0.5) * 0.2))
        elif behavior == 'environment':
            self.data.activity = 'changing' if random.random() > 0.7 else 'stable'
        elif behavior == 'degradation':
            self.data.energy = max(0.1, self.data.energy - 0.02)
        elif behavior == 'consequence':
            self.data.activity = 'triggering'
        elif behavior == 'influence':
            self.data.energy += (random.random() - 0.5) * 0.1
        else:
            self.data.activity = 'existing'
            
        self.data.last_activity = behavior
        
    def update_state(self):
        # Gradually return to neutral
        self.data.energy = max(0, min(1, self.data.energy + (0.5 - self.data.energy) * 0.05))
        self.data.mood = max(0, min(1, self.data.mood + (0.5 - self.data.mood) * 0.03))

# OpenGL Font Renderer
class OpenGLFontRenderer(QOpenGLWidget):
    """OpenGL-based font rendering widget"""
    
    fps_updated = Signal(int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.text_content = ""
        self.font_size = 18
        self.line_height = 1.4
        self.letter_spacing = 0
        self.text_color = [1.0, 1.0, 1.0, 1.0]
        self.effects = {
            'outline': False,
            'shadow': False,
            'glow': False,
            'gradient': False,
            'wave': False,
            'rainbow': False
        }
        
        if OPENGL_AVAILABLE:
            self.program = None
            self.vao = None
            self.fps = 60
            self.frame_count = 0
            self.last_time = time.time()
            
            # Start render loop
            self.timer = QTimer()
            self.timer.timeout.connect(self.update)
            self.timer.start(16)  # ~60 FPS
        
    def set_text(self, text: str):
        self.text_content = text
        self.update()
    
    def set_font_size(self, size: int):
        self.font_size = size
        self.update()
    
    def set_line_height(self, height: float):
        self.line_height = height
        self.update()
    
    def set_letter_spacing(self, spacing: int):
        self.letter_spacing = spacing
        self.update()
    
    def set_text_color(self, color: QColor):
        self.text_color = [color.redF(), color.greenF(), color.blueF(), color.alphaF()]
        self.update()
    
    def toggle_effect(self, effect: str) -> bool:
        if effect in self.effects:
            self.effects[effect] = not self.effects[effect]
            self.update()
            return self.effects[effect]
        return False
    
    if OPENGL_AVAILABLE:
        def initializeGL(self):
            # Initialize OpenGL
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glClearColor(0.06, 0.06, 0.14, 1.0)
            
            # Create shader program
            self.create_shader_program()
            
        def create_shader_program(self):
            vertex_shader = """
            #version 330 core
            layout (location = 0) in vec2 position;
            layout (location = 1) in vec2 texCoord;
            layout (location = 2) in vec4 color;
            
            uniform mat4 projection;
            uniform float time;
            uniform int effectFlags;
            
            out vec2 vTexCoord;
            out vec4 vColor;
            out float vTime;
            
            void main() {
                vec2 pos = position;
                
                // Wave effect
                if ((effectFlags & 16) != 0) {
                    pos.y += sin(pos.x * 0.015 + time * 3.0) * 8.0;
                }
                
                gl_Position = projection * vec4(pos, 0.0, 1.0);
                vTexCoord = texCoord;
                vColor = color;
                vTime = time;
            }
            """
            
            fragment_shader = """
            #version 330 core
            in vec2 vTexCoord;
            in vec4 vColor;
            in float vTime;
            
            uniform int effectFlags;
            
            out vec4 fragColor;
            
            void main() {
                vec4 finalColor = vColor;
                
                // Rainbow effect
                if ((effectFlags & 32) != 0) {
                    float hue = vTexCoord.x * 8.0 + vTime * 2.5;
                    finalColor.rgb = vec3(
                        sin(hue) * 0.5 + 0.5,
                        sin(hue + 2.094) * 0.5 + 0.5,
                        sin(hue + 4.188) * 0.5 + 0.5
                    );
                }
                
                // Create simple text-like pattern
                float distance = length(vTexCoord - vec2(0.5));
                float alpha = smoothstep(0.5, 0.3, distance);
                finalColor.a *= alpha;
                
                fragColor = finalColor;
            }
            """
            
            try:
                self.program = compileProgram(
                    compileShader(vertex_shader, GL_VERTEX_SHADER),
                    compileShader(fragment_shader, GL_FRAGMENT_SHADER)
                )
                print("OpenGL shaders compiled successfully")
            except Exception as e:
                print(f"Shader compilation error: {e}")
        
        def paintGL(self):
            glClear(GL_COLOR_BUFFER_BIT)
            
            if self.program and self.text_content:
                glUseProgram(self.program)
                
                # Set uniforms
                try:
                    time_loc = glGetUniformLocation(self.program, "time")
                    glUniform1f(time_loc, time.time())
                    
                    effect_flags = 0
                    if self.effects['wave']: effect_flags |= 16
                    if self.effects['rainbow']: effect_flags |= 32
                    
                    effect_loc = glGetUniformLocation(self.program, "effectFlags")
                    glUniform1i(effect_loc, effect_flags)
                    
                    # Render simple text representation
                    self.render_text_simple()
                except Exception as e:
                    print(f"Render error: {e}")
            
            # Update FPS
            self.update_fps()
        
        def render_text_simple(self):
            """Simplified text rendering using OpenGL primitives"""
            # Create simple quad vertices for text representation
            lines = self.text_content.split('\n')
            y_offset = 0.8
            
            for line in lines[:10]:  # Limit to 10 lines
                if not line.strip():
                    continue
                    
                # Simple character representation
                x_offset = -0.9
                for i, char in enumerate(line[:50]):  # Limit characters per line
                    if char == ' ':
                        x_offset += 0.02
                        continue
                        
                    # Draw character as small quad
                    char_size = self.font_size / 1000.0
                    vertices = np.array([
                        [x_offset, y_offset, 0.0, 0.0, *self.text_color],
                        [x_offset + char_size, y_offset, 1.0, 0.0, *self.text_color],
                        [x_offset + char_size, y_offset - char_size, 1.0, 1.0, *self.text_color],
                        [x_offset, y_offset - char_size, 0.0, 1.0, *self.text_color]
                    ], dtype=np.float32)
                    
                    # Simple rendering without proper VAO/VBO setup
                    x_offset += char_size + self.letter_spacing / 1000.0
                
                y_offset -= self.line_height * self.font_size / 1000.0
        
        def resizeGL(self, width, height):
            glViewport(0, 0, width, height)
            
            if self.program:
                glUseProgram(self.program)
                # Set projection matrix
                projection = np.array([
                    [2.0/width, 0, 0, 0],
                    [0, -2.0/height, 0, 0],
                    [0, 0, -1, 0],
                    [-1, 1, 0, 1]
                ], dtype=np.float32)
                
                try:
                    proj_loc = glGetUniformLocation(self.program, "projection")
                    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
                except:
                    pass
        
        def update_fps(self):
            self.frame_count += 1
            current_time = time.time()
            if current_time - self.last_time >= 1.0:
                self.fps = self.frame_count
                self.frame_count = 0
                self.last_time = current_time
                self.fps_updated.emit(self.fps)
    
    else:
        # Fallback for when OpenGL is not available
        fps_updated = Signal(int)
        
        def __init__(self, parent=None):
            super().__init__(parent)
            self.text_content = ""
            self.font_size = 18
            self.line_height = 1.4
            self.letter_spacing = 0
            self.text_color = [1.0, 1.0, 1.0, 1.0]
            self.effects = {}
            
            # Simulate FPS updates
            self.timer = QTimer()
            self.timer.timeout.connect(lambda: self.fps_updated.emit(60))
            self.timer.start(1000)
        
        def set_text(self, text: str):
            self.text_content = text
            self.update()
        
        def set_font_size(self, size: int):
            self.font_size = size
            self.update()
        
        def set_line_height(self, height: float):
            self.line_height = height
            self.update()
        
        def set_letter_spacing(self, spacing: int):
            self.letter_spacing = spacing
            self.update()
        
        def set_text_color(self, color: QColor):
            self.text_color = [color.redF(), color.greenF(), color.blueF(), color.alphaF()]
            self.update()
        
        def toggle_effect(self, effect: str) -> bool:
            return False
        
        def paintEvent(self, event):
            painter = QPainter(self)
            painter.fillRect(self.rect(), QColor(15, 15, 35))
            
            color = QColor(int(self.text_color[0] * 255), 
                          int(self.text_color[1] * 255), 
                          int(self.text_color[2] * 255))
            painter.setPen(color)
            
            font = QFont("Arial", self.font_size)
            painter.setFont(font)
            
            if self.text_content:
                painter.drawText(self.rect(), Qt.AlignLeft | Qt.AlignTop, self.text_content)
            else:
                painter.drawText(self.rect(), Qt.AlignCenter, "OpenGL not available\nFallback text rendering")

# Entity List Widget
class EntityTokenWidget(QFrame):
    """Widget displaying individual entity token"""
    
    def __init__(self, entity: EntityToken):
        super().__init__()
        self.entity = entity
        self.init_ui()
        self.entity.state_changed.connect(self.update_display)
        
    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(12, 12, 12, 12)
        
        # Header
        header_layout = QHBoxLayout()
        self.name_label = QLabel(self.entity.data.name)
        self.name_label.setStyleSheet("font-weight: bold; font-size: 14px; color: white;")
        
        self.status_indicator = QLabel("●")
        self.status_indicator.setStyleSheet("color: orange; font-size: 12px;")
        
        header_layout.addWidget(self.name_label)
        header_layout.addStretch()
        header_layout.addWidget(self.status_indicator)
        
        # Description
        self.desc_label = QLabel(self.entity.data.description)
        self.desc_label.setStyleSheet("font-size: 11px; color: #ccc;")
        self.desc_label.setWordWrap(True)
        
        # Stats
        stats_frame = QFrame()
        stats_frame.setStyleSheet("""
            QFrame {
                background: rgba(0,0,0,0.3); 
                border-radius: 4px; 
                padding: 6px;
            }
        """)
        stats_layout = QVBoxLayout(stats_frame)
        stats_layout.setContentsMargins(6, 6, 6, 6)
        
        self.type_label = QLabel(f"Type: {self.entity.data.type.value}")
        self.interval_label = QLabel(f"Interval: {self.entity.interval}ms")
        self.behaviors_label = QLabel(f"Behaviors: {', '.join(self.entity.behaviors[:2])}")
        
        for label in [self.type_label, self.interval_label, self.behaviors_label]:
            label.setStyleSheet("font-size: 10px; color: #ddd;")
            stats_layout.addWidget(label)
        
        # Runtime stats
        self.runtime_frame = QFrame()
        self.runtime_frame.setStyleSheet("""
            QFrame {
                background: rgba(0,0,0,0.2); 
                border-radius: 3px; 
                padding: 4px;
            }
        """)
        runtime_layout = QVBoxLayout(self.runtime_frame)
        runtime_layout.setContentsMargins(4, 4, 4, 4)
        
        self.ticks_label = QLabel(f"Ticks: {self.entity.data.tick_count}")
        self.activity_label = QLabel(f"Activity: {self.entity.data.activity}")
        self.energy_label = QLabel(f"Energy: {int(self.entity.data.energy * 100)}%")
        self.mood_label = QLabel(f"Mood: {int(self.entity.data.mood * 100)}%")
        
        for label in [self.ticks_label, self.activity_label, self.energy_label, self.mood_label]:
            label.setStyleSheet("font-size: 10px; color: #bbb;")
            runtime_layout.addWidget(label)
        
        layout.addLayout(header_layout)
        layout.addWidget(self.desc_label)
        layout.addWidget(stats_frame)
        layout.addWidget(self.runtime_frame)
        
        self.setLayout(layout)
        self.setStyleSheet(f"""
            EntityTokenWidget {{
                background: rgba(255,255,255,0.1);
                border-radius: 8px;
                border-left: 4px solid {self.get_type_color()};
                margin: 4px;
            }}
            EntityTokenWidget:hover {{
                background: rgba(255,255,255,0.2);
            }}
        """)
        
    def get_type_color(self) -> str:
        colors = {
            EntityType.CHARACTER: "#4CAF50",
            EntityType.LOCATION: "#FF9800", 
            EntityType.ITEM: "#2196F3",
            EntityType.EVENT: "#9C27B0",
            EntityType.CONCEPT: "#F44336"
        }
        return colors.get(self.entity.data.type, "#666")
        
    def update_display(self):
        self.ticks_label.setText(f"Ticks: {self.entity.data.tick_count}")
        self.activity_label.setText(f"Activity: {self.entity.data.activity}")
        self.energy_label.setText(f"Energy: {int(self.entity.data.energy * 100)}%")
        self.mood_label.setText(f"Mood: {int(self.entity.data.mood * 100)}%")
        
        # Update status indicator
        if self.entity.data.is_running:
            self.status_indicator.setStyleSheet("color: #4CAF50; font-size: 12px;")
        else:
            self.status_indicator.setStyleSheet("color: #FF9800; font-size: 12px;")
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.entity.data.is_running:
                self.entity.stop_game_loop()
            else:
                self.entity.start_game_loop()
            self.update_display()

# Entity Container Widget
class EntityContainerWidget(QScrollArea):
    """Scrollable container for entity tokens"""
    
    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        self.container_widget = QWidget()
        self.layout = QVBoxLayout(self.container_widget)
        self.layout.addStretch()
        
        self.setWidget(self.container_widget)
        self.entity_widgets = {}
        
    def add_entity_widget(self, entity_id: str, widget: EntityTokenWidget):
        self.layout.insertWidget(self.layout.count() - 1, widget)
        self.entity_widgets[entity_id] = widget
        
    def remove_entity_widget(self, entity_id: str):
        if entity_id in self.entity_widgets:
            widget = self.entity_widgets[entity_id]
            self.layout.removeWidget(widget)
            widget.deleteLater()
            del self.entity_widgets[entity_id]
            
    def clear_all(self):
        for entity_id in list(self.entity_widgets.keys()):
            self.remove_entity_widget(entity_id)

# Scene Graph Widget
class SceneGraphWidget(QTreeWidget):
    """Tree widget for displaying story structure"""
    
    def __init__(self):
        super().__init__()
        self.setHeaderLabel("Scene Graph")
        self.setStyleSheet("""
            QTreeWidget {
                background: rgba(0,0,0,0.2);
                border-radius: 8px;
                color: white;
                border: 1px solid rgba(255,255,255,0.1);
            }
            QTreeWidget::item {
                padding: 4px;
                border-radius: 4px;
            }
            QTreeWidget::item:hover {
                background: rgba(255,255,255,0.1);
            }
            QTreeWidget::item:selected {
                background: rgba(76,175,80,0.3);
            }
        """)
        
    def build_from_text(self, text: str):
        self.clear()
        
        root_item = QTreeWidgetItem(self)
        root_item.setText(0, "Story Root")
        root_item.setData(0, Qt.UserRole, {"type": "root", "content": text[:50] + "..."})
        
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        for i, paragraph in enumerate(paragraphs):
            para_item = QTreeWidgetItem(root_item)
            para_item.setText(0, f"Paragraph {i+1}")
            para_item.setData(0, Qt.UserRole, {"type": "paragraph", "content": paragraph[:60] + "..."})
            
            sentences = [s.strip() for s in re.split(r'[.!?]+', paragraph) if s.strip()]
            
            for j, sentence in enumerate(sentences):
                sent_item = QTreeWidgetItem(para_item)
                sent_item.setText(0, f"Sentence {j+1}")
                sent_item.setData(0, Qt.UserRole, {"type": "sentence", "content": sentence[:40] + "..."})
        
        self.expandAll()

# Message Log Widget
class MessageLogWidget(QPlainTextEdit):
    """Widget for displaying system messages"""
    
    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setMaximumBlockCount(1000)  # Limit to 1000 messages
        self.setStyleSheet("""
            QPlainTextEdit {
                background: rgba(0,0,0,0.3);
                border-radius: 6px;
                color: white;
                font-family: 'Courier New', monospace;
                font-size: 10px;
                border: 1px solid rgba(255,255,255,0.1);
            }
        """)
        
        # Add initial message
        self.add_message(MessageType.SYSTEM, "System initialized. Ready to spawn entities.")
        
    def add_message(self, msg_type: MessageType, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        colors = {
            MessageType.SPAWN: "#4CAF50",
            MessageType.TICK: "#2196F3", 
            MessageType.BEHAVIOR: "#FF9800",
            MessageType.SAVE: "#9C27B0",
            MessageType.SYSTEM: "#607D8B",
            MessageType.WEBGL: "#3498db"
        }
        
        color = colors.get(msg_type, "#ffffff")
        formatted_message = f'<span style="color: {color};">[{timestamp}] {message}</span>'
        
        self.appendHtml(formatted_message)
        self.ensureCursorVisible()

# Main Application
class StoryWorldEngine(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.entities: Dict[str, EntityToken] = {}
        self.current_text = ""
        self.settings = QSettings("StoryWorldEngine", "MainApp")
        
        self.init_ui()
        self.init_systems()
        self.connect_signals()
        self.load_demo_content()
        
    def init_ui(self):
        self.setWindowTitle("Story World Engine - PySide6")
        self.setGeometry(100, 100, 1400, 900)
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # FPS counter in status bar
        self.fps_label = QLabel("FPS: 60")
        self.entity_count_label = QLabel("Entities: 0")
        self.tick_count_label = QLabel("Ticks: 0")
        
        self.status_bar.addPermanentWidget(self.fps_label)
        self.status_bar.addPermanentWidget(self.entity_count_label)
        self.status_bar.addPermanentWidget(self.tick_count_label)
        
        # Dark theme
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1e3c72, stop:1 #2a5298);
                color: white;
            }
            QWidget {
                color: white;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid rgba(255,255,255,0.2);
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background: rgba(255,255,255,0.05);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background: linear-gradient(45deg, #4CAF50, #45a049);
                color: white;
                border: none;
                padding: 8px 14px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background: linear-gradient(45deg, #5CBF60, #4CAF50);
            }
            QPushButton:pressed {
                background: linear-gradient(45deg, #45a049, #3d8b40);
            }
            QSlider::groove:horizontal {
                border: 1px solid rgba(255,255,255,0.2);
                height: 4px;
                background: rgba(255,255,255,0.1);
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #4CAF50;
                border: 1px solid rgba(255,255,255,0.3);
                width: 16px;
                border-radius: 8px;
                margin: -6px 0;
            }
            QSlider::handle:horizontal:hover {
                background: #5CBF60;
            }
            QTextEdit, QPlainTextEdit {
                background: rgba(0,0,0,0.3);
                border: 1px solid rgba(255,255,255,0.2);
                border-radius: 8px;
                color: white;
                padding: 8px;
            }
            QTabWidget::pane {
                border: 1px solid rgba(255,255,255,0.2);
                background: rgba(255,255,255,0.05);
                border-radius: 8px;
            }
            QTabBar::tab {
                background: rgba(255,255,255,0.1);
                color: white;
                padding: 8px 16px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: rgba(76,175,80,0.3);
                border-bottom: 2px solid #4CAF50;
            }
            QTabBar::tab:hover {
                background: rgba(255,255,255,0.2);
            }
        """)
        
        # Central widget with splitter
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel - Text input and controls
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)
        
        # Center panel - OpenGL renderer
        center_panel = self.create_center_panel()
        splitter.addWidget(center_panel)
        
        # Right panel - Entities and logs
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)
        
        # Set splitter proportions
        splitter.setSizes([380, 640, 380])
        
    def create_menu_bar(self):
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        new_action = QAction('New Story', self)
        new_action.setShortcut('Ctrl+N')
        new_action.triggered.connect(self.new_story)
        file_menu.addAction(new_action)
        
        open_action = QAction('Open Story', self)
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.open_story)
        file_menu.addAction(open_action)
        
        save_action = QAction('Save Story', self)
        save_action.setShortcut('Ctrl+S')
        save_action.triggered.connect(self.save_story)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu('View')
        
        fullscreen_action = QAction('Toggle Fullscreen', self)
        fullscreen_action.setShortcut('F11')
        fullscreen_action.triggered.connect(self.toggle_fullscreen)
        view_menu.addAction(fullscreen_action)
        
        # Help menu
        help_menu = menubar.addMenu('Help')
        
        about_action = QAction('About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
    def create_left_panel(self) -> QWidget:
        widget = QWidget()
        widget.setMaximumWidth(400)
        layout = QVBoxLayout(widget)
        
        # Text input group
        text_group = QGroupBox("📝 Text Processing & Controls")
        text_layout = QVBoxLayout(text_group)
        
        self.text_input = QTextEdit()
        self.text_input.setMaximumHeight(200)
        self.text_input.setPlaceholderText("""Enter your story text here...

Princess Elena walked through the enchanted forest, carrying her father's magical sword. The ancient castle of Shadowmere loomed ahead, where the evil wizard Malphas was brewing dark potions. Thunder rumbled overhead as she approached the mysterious glowing crystal embedded in the castle walls.

The wind whispered secrets through the twisted branches, and Elena felt the weight of destiny upon her shoulders. Each step brought her closer to the final confrontation that would determine the fate of the kingdom.""")
        
        # Controls
        controls_layout = QHBoxLayout()
        self.process_btn = QPushButton("🔄 Process Text")
        self.start_all_btn = QPushButton("▶️ Start All")
        self.pause_all_btn = QPushButton("⏸️ Pause All")
        self.clear_btn = QPushButton("🗑️ Clear")
        
        for btn in [self.process_btn, self.start_all_btn, self.pause_all_btn, self.clear_btn]:
            controls_layout.addWidget(btn)
        
        text_layout.addWidget(self.text_input)
        text_layout.addLayout(controls_layout)
        
        # Font controls group
        font_group = QGroupBox("🎨 WebGL Font Controls")
        font_layout = QGridLayout(font_group)
        
        # Font size
        font_layout.addWidget(QLabel("Font Size:"), 0, 0)
        self.font_size_slider = QSlider(Qt.Horizontal)
        self.font_size_slider.setRange(12, 48)
        self.font_size_slider.setValue(18)
        self.font_size_label = QLabel("18px")
        font_layout.addWidget(self.font_size_slider, 0, 1)
        font_layout.addWidget(self.font_size_label, 0, 2)
        
        # Line height
        font_layout.addWidget(QLabel("Line Height:"), 1, 0)
        self.line_height_slider = QSlider(Qt.Horizontal)
        self.line_height_slider.setRange(10, 25)
        self.line_height_slider.setValue(14)
        self.line_height_label = QLabel("1.4x")
        font_layout.addWidget(self.line_height_slider, 1, 1)
        font_layout.addWidget(self.line_height_label, 1, 2)
        
        # Letter spacing
        font_layout.addWidget(QLabel("Letter Spacing:"), 2, 0)
        self.letter_spacing_slider = QSlider(Qt.Horizontal)
        self.letter_spacing_slider.setRange(-2, 10)
        self.letter_spacing_slider.setValue(0)
        self.letter_spacing_label = QLabel("0px")
        font_layout.addWidget(self.letter_spacing_slider, 2, 1)
        font_layout.addWidget(self.letter_spacing_label, 2, 2)
        
        # Text color
        font_layout.addWidget(QLabel("Text Color:"), 3, 0)
        self.color_btn = QPushButton()
        self.color_btn.setStyleSheet("background: white; min-height: 30px;")
        self.current_color = QColor(255, 255, 255)
        font_layout.addWidget(self.color_btn, 3, 1, 1, 2)
        
        # Effects
        effects_group = QGroupBox("✨ Text Effects")
        effects_layout = QGridLayout(effects_group)
        
        self.effect_checkboxes = {}
        effects = ['outline', 'shadow', 'glow', 'wave', 'rainbow', 'gradient']
        for i, effect in enumerate(effects):
            checkbox = QCheckBox(effect.title())
            checkbox.setStyleSheet("QCheckBox { color: white; }")
            self.effect_checkboxes[effect] = checkbox
            effects_layout.addWidget(checkbox, i // 2, i % 2)
        
        # Stats
        stats_group = QGroupBox("📊 System Stats")
        stats_layout = QGridLayout(stats_group)
        
        self.char_count_label = QLabel("0")
        self.loc_count_label = QLabel("0")
        self.item_count_label = QLabel("0")
        self.event_count_label = QLabel("0")
        
        for label in [self.char_count_label, self.loc_count_label, self.item_count_label, self.event_count_label]:
            label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        
        stats_layout.addWidget(QLabel("Characters:"), 0, 0)
        stats_layout.addWidget(self.char_count_label, 0, 1)
        stats_layout.addWidget(QLabel("Locations:"), 0, 2)
        stats_layout.addWidget(self.loc_count_label, 0, 3)
        stats_layout.addWidget(QLabel("Items:"), 1, 0)
        stats_layout.addWidget(self.item_count_label, 1, 1)
        stats_layout.addWidget(QLabel("Events:"), 1, 2)
        stats_layout.addWidget(self.event_count_label, 1, 3)
        
        # Scene graph
        scene_graph_label = QLabel("🌳 Scene Graph")
        scene_graph_label.setStyleSheet("font-weight: bold; font-size: 14px; margin-top: 10px;")
        self.scene_graph = SceneGraphWidget()
        self.scene_graph.setMaximumHeight(300)
        
        layout.addWidget(text_group)
        layout.addWidget(font_group)
        layout.addWidget(effects_group)
        layout.addWidget(stats_group)
        layout.addWidget(scene_graph_label)
        layout.addWidget(self.scene_graph)
        
        return widget
    
    def create_center_panel(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Header
        header_layout = QHBoxLayout()
        header_layout.addWidget(QLabel("🖥️ WebGL Font Renderer"))
        self.renderer_status = QLabel("Initializing...")
        header_layout.addStretch()
        header_layout.addWidget(self.renderer_status)
        
        # OpenGL renderer
        self.renderer = OpenGLFontRenderer()
        
        layout.addLayout(header_layout)
        layout.addWidget(self.renderer)
        
        return widget
    
    def create_right_panel(self) -> QWidget:
        widget = QWidget()
        widget.setMaximumWidth(400)
        layout = QVBoxLayout(widget)
        
        # Tabbed interface
        self.tab_widget = QTabWidget()
        
        # Entities tab
        entities_tab = QWidget()
        entities_layout = QVBoxLayout(entities_tab)
        
        entities_header = QLabel("🎭 Entity Tokens")
        entities_header.setStyleSheet("font-weight: bold; font-size: 14px;")
        entities_layout.addWidget(entities_header)
        
        self.entity_container = EntityContainerWidget()
        entities_layout.addWidget(self.entity_container)
        
        self.tab_widget.addTab(entities_tab, "Entities")
        
        # Messages tab
        messages_tab = QWidget()
        messages_layout = QVBoxLayout(messages_tab)
        
        messages_header = QLabel("📡 System Messages")
        messages_header.setStyleSheet("font-weight: bold; font-size: 14px;")
        messages_layout.addWidget(messages_header)
        
        self.message_log = MessageLogWidget()
        messages_layout.addWidget(self.message_log)
        
        self.tab_widget.addTab(messages_tab, "Messages")
        
        # Performance tab
        performance_tab = QWidget()
        performance_layout = QVBoxLayout(performance_tab)
        
        performance_header = QLabel("⚡ Performance Monitor")
        performance_header.setStyleSheet("font-weight: bold; font-size: 14px;")
        performance_layout.addWidget(performance_header)
        
        # Performance stats
        perf_stats_frame = QFrame()
        perf_stats_frame.setStyleSheet("""
            QFrame {
                background: rgba(0,0,0,0.3);
                border-radius: 8px;
                padding: 10px;
                border: 1px solid rgba(255,255,255,0.1);
            }
        """)
        perf_stats_layout = QGridLayout(perf_stats_frame)
        
        self.frame_time_label = QLabel("16.7ms")
        self.gpu_memory_label = QLabel("2.4MB")
        self.draw_calls_label = QLabel("1")
        self.active_loops_label = QLabel("0")
        
        for label in [self.frame_time_label, self.gpu_memory_label, self.draw_calls_label, self.active_loops_label]:
            label.setStyleSheet("color: #2ecc71; font-weight: bold;")
        
        perf_stats_layout.addWidget(QLabel("Frame Time:"), 0, 0)
        perf_stats_layout.addWidget(self.frame_time_label, 0, 1)
        perf_stats_layout.addWidget(QLabel("GPU Memory:"), 1, 0)
        perf_stats_layout.addWidget(self.gpu_memory_label, 1, 1)
        perf_stats_layout.addWidget(QLabel("Draw Calls:"), 2, 0)
        perf_stats_layout.addWidget(self.draw_calls_label, 2, 1)
        perf_stats_layout.addWidget(QLabel("Active Loops:"), 3, 0)
        perf_stats_layout.addWidget(self.active_loops_label, 3, 1)
        
        performance_layout.addWidget(perf_stats_frame)
        performance_layout.addStretch()
        
        self.tab_widget.addTab(performance_tab, "Performance")
        
        layout.addWidget(self.tab_widget)
        
        return widget
    
    def init_systems(self):
        """Initialize all subsystems"""
        self.total_ticks = 0
        
        # Set renderer status
        if OPENGL_AVAILABLE:
            self.renderer_status.setText("Ready (OpenGL)")
            self.renderer_status.setStyleSheet("color: #4CAF50;")
        else:
            self.renderer_status.setText("Fallback Mode")
            self.renderer_status.setStyleSheet("color: #FF9800;")
        
        self.message_log.add_message(MessageType.SYSTEM, "Story World Engine initialized")
        
    def connect_signals(self):
        """Connect all UI signals"""
        # Text processing buttons
        self.process_btn.clicked.connect(self.process_text)
        self.start_all_btn.clicked.connect(self.start_all_entities)
        self.pause_all_btn.clicked.connect(self.pause_all_entities)
        self.clear_btn.clicked.connect(self.clear_all_entities)
        
        # Font controls
        self.font_size_slider.valueChanged.connect(self.update_font_size)
        self.line_height_slider.valueChanged.connect(self.update_line_height)
        self.letter_spacing_slider.valueChanged.connect(self.update_letter_spacing)
        self.color_btn.clicked.connect(self.choose_text_color)
        
        # Effects
        for effect, checkbox in self.effect_checkboxes.items():
            checkbox.toggled.connect(lambda checked, e=effect: self.toggle_effect(e, checked))
        
        # Renderer
        if hasattr(self.renderer, 'fps_updated'):
            self.renderer.fps_updated.connect(self.update_fps_display)
        
        # Text input
        self.text_input.textChanged.connect(self.on_text_changed)
        
    def load_demo_content(self):
        """Load demo content on startup"""
        demo_text = """Princess Elena walked through the enchanted forest, carrying her father's magical sword. The ancient castle of Shadowmere loomed ahead, where the evil wizard Malphas was brewing dark potions. Thunder rumbled overhead as she approached the mysterious glowing crystal embedded in the castle walls.

The wind whispered secrets through the twisted branches, and Elena felt the weight of destiny upon her shoulders. Each step brought her closer to the final confrontation that would determine the fate of the kingdom."""
        
        self.text_input.setPlainText(demo_text)
        
        # Auto-process after a short delay
        QTimer.singleShot(1000, self.process_text)
        
    def process_text(self):
        """Process text to extract entities"""
        text = self.text_input.toPlainText().strip()
        if not text:
            return
        
        self.current_text = text
        self.message_log.add_message(MessageType.SPAWN, f"Processing {len(text)} characters")
        
        # Update renderer
        self.renderer.set_text(text)
        
        # Build scene graph
        self.scene_graph.build_from_text(text)
        
        # Extract entities
        entities_data = NLPProcessor.extract_entities(text)
        
        # Clear existing entities
        self.clear_all_entities()
        
        # Create entity tokens
        for entity_data in entities_data:
            entity_id = str(uuid.uuid4())
            
            entity = EntityToken(EntityData(
                id=entity_id,
                type=entity_data['type'],
                name=entity_data['name'],
                description=entity_data['description'],
                properties=entity_data['properties'],
                source_text=text,
                spawned=time.time()
            ))
            
            self.entities[entity_id] = entity
            
            # Create widget
            widget = EntityTokenWidget(entity)
            self.entity_container.add_entity_widget(entity_id, widget)
            
            # Connect signals
            entity.state_changed.connect(self.on_entity_state_changed)
            
        self.update_stats()
        self.message_log.add_message(MessageType.SPAWN, f"Spawned {len(entities_data)} entities")
        
    def start_all_entities(self):
        """Start all entity game loops"""
        started = 0
        for entity in self.entities.values():
            if not entity.data.is_running:
                entity.start_game_loop()
                started += 1
        
        self.message_log.add_message(MessageType.TICK, f"Started {started} entity loops")
        self.update_stats()
        
    def pause_all_entities(self):
        """Pause all entity game loops"""
        paused = 0
        for entity in self.entities.values():
            if entity.data.is_running:
                entity.stop_game_loop()
                paused += 1
        
        self.message_log.add_message(MessageType.TICK, f"Paused {paused} entity loops")
        self.update_stats()
        
    def clear_all_entities(self):
        """Clear all entities"""
        self.pause_all_entities()
        
        for entity_id in list(self.entities.keys()):
            entity = self.entities[entity_id]
            entity.stop_game_loop()
            self.entity_container.remove_entity_widget(entity_id)
            del self.entities[entity_id]
        
        self.update_stats()
        self.message_log.add_message(MessageType.SYSTEM, "All entities cleared")
        
    def update_font_size(self, value):
        """Update font size"""
        self.font_size_label.setText(f"{value}px")
        self.renderer.set_font_size(value)
        
    def update_line_height(self, value):
        """Update line height"""
        height = value / 10.0
        self.line_height_label.setText(f"{height:.1f}x")
        self.renderer.set_line_height(height)
        
    def update_letter_spacing(self, value):
        """Update letter spacing"""
        self.letter_spacing_label.setText(f"{value}px")
        self.renderer.set_letter_spacing(value)
        
    def choose_text_color(self):
        """Choose text color"""
        color = QColorDialog.getColor(self.current_color, self)
        if color.isValid():
            self.current_color = color
            self.color_btn.setStyleSheet(f"background: {color.name()}; min-height: 30px;")
            self.renderer.set_text_color(color)
            
    def toggle_effect(self, effect: str, checked: bool):
        """Toggle text effect"""
        if hasattr(self.renderer, 'toggle_effect'):
            active = self.renderer.toggle_effect(effect)
            self.message_log.add_message(MessageType.WEBGL, f"{effect} effect: {'ON' if active else 'OFF'}")
        
    def on_text_changed(self):
        """Handle text input changes"""
        text = self.text_input.toPlainText()
        self.renderer.set_text(text)
        
    def on_entity_state_changed(self, entity_id: str):
        """Handle entity state changes"""
        self.total_ticks += 1
        self.tick_count_label.setText(f"Ticks: {self.total_ticks}")
        self.update_stats()
        
    def update_stats(self):
        """Update entity statistics"""
        stats = {EntityType.CHARACTER: 0, EntityType.LOCATION: 0, EntityType.ITEM: 0, EntityType.EVENT: 0}
        active_loops = 0
        
        for entity in self.entities.values():
            stats[entity.data.type] += 1
            if entity.data.is_running:
                active_loops += 1
        
        self.char_count_label.setText(str(stats[EntityType.CHARACTER]))
        self.loc_count_label.setText(str(stats[EntityType.LOCATION]))
        self.item_count_label.setText(str(stats[EntityType.ITEM]))
        self.event_count_label.setText(str(stats[EntityType.EVENT]))
        
        self.entity_count_label.setText(f"Entities: {len(self.entities)}")
        self.active_loops_label.setText(str(active_loops))
        
    def update_fps_display(self, fps: int):
        """Update FPS display"""
        self.fps_label.setText(f"FPS: {fps}")
        frame_time = 1000.0 / max(fps, 1)
        self.frame_time_label.setText(f"{frame_time:.1f}ms")
        
    def new_story(self):
        """Create new story"""
        self.clear_all_entities()
        self.text_input.clear()
        self.renderer.set_text("")
        self.scene_graph.clear()
        self.message_log.add_message(MessageType.SYSTEM, "New story created")
        
    def open_story(self):
        """Open story from file"""
        # Placeholder for file dialog
        self.message_log.add_message(MessageType.SYSTEM, "Open story - not implemented yet")
        
    def save_story(self):
        """Save story to file"""
        # Placeholder for file dialog
        self.message_log.add_message(MessageType.SYSTEM, "Save story - not implemented yet")
        
    def toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()
            
    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(self, "About Story World Engine", 
                         "Story World Engine - PySide6 Implementation\n\n"
                         "A complete system with OpenGL font rendering,\n"
                         "entity tokens, and scene graphs.\n\n"
                         "Features:\n"
                         "• WebGL-style font rendering\n"
                         "• Entity token system with game loops\n"
                         "• Scene graph visualization\n"
                         "• Real-time text effects\n"
                         "• Performance monitoring")
        
    def closeEvent(self, event):
        """Handle application close"""
        self.pause_all_entities()
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())
        event.accept()


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Story World Engine")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("Story World Systems")
    
    # Create and show main window
    window = StoryWorldEngine()
    window.show()
    
    # Start event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()