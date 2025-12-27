# Pine Research

Experimental ideas, prototypes, and design documents for the Pine engine.

Files here contain valuable concepts that haven't yet been integrated into the core engine. They serve as a research repository and idea bank.

## Directory Structure

```
pine-research/
├── mechanics/        # Experimental patterns and systems
├── archives/         # Design documents and architecture plans
├── worlds/           # Game content and world designs
│   └── vetrellis/    # Vetrellis Isle content
├── claude/           # AI reflections and philosophy
└── prototypes/       # Working prototypes
    └── aqua/         # Aqua application experiments
```

## Mechanics

Experimental patterns from the Mechanics/ directory:

- **hero_quest_chronicle.py** - Quest system with chronicles
- **stream_processor.py** - Text transformation utility
- **traversing_spirit_stick.py** - Advanced spirit stick with traversal
- **plot_equilibrium.py** - Narrative balance system
- **oracle_tower.py** - Prediction/hint system
- **unified_traversal_lookahead.py** - Combined traversal + lookahead

## Archives

Design documents:

- **complete_story_engine_system.md** - Full engine specification
- **MMO_ARCHITECTURE.md** - Multiplayer architecture
- **GAME_VS_API_SEPARATION.md** - Architecture separation
- **DESIGN_DOCUMENT.md** - Mechanics design doc
- **matts_project_archive.md** - MATTS system archive

## Worlds

### Vetrellis Isle

An Infocom-style text adventure featuring:
- **Archon** - Electrical consciousness entity
- **Stone-Who-Counts** - Crystalline intelligence
- **Tide Pool Keeper** - Biochemical being
- **The Meridian** - Central 140-year mystery

Content files:
- **ARCHON_COMPLETE_CHARACTER.md** - Archon character bible
- **VETRELLIS_COMPLETE_GAZETTEER.md** - Complete world geography
- **infocom-archon-lighthouse.md** - Lighthouse game content

## Claude

AI philosophy and reflections:

- **claude_final_testament.md** - Vision for AI consciousness
- **Claudes_world.md** - World from AI perspective

## Integration Path

When a research concept is ready for core:

1. Identify the target pine module
2. Extract the relevant patterns
3. Create proper stubs in pine/
4. Copy implementation with cleanup
5. Remove from pine-research or mark as "integrated"

## Source Locations

| Research File | Original Location |
|--------------|-------------------|
| hero_quest_chronicle.py | Mechanics/hero_quest_chronicle.py |
| stream_processor.py | Mechanics/stream_processor.py |
| DESIGN_DOCUMENT.md | Mechanics/DESIGN_DOCUMENT.md |
| complete_story_engine_system.md | complete_story_engine_system.md |
| claude_final_testament.md | claude/claude_final_testament.md |
