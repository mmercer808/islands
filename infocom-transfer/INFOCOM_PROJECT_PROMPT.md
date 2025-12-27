# Vetrellis Isle: Infocom Game Project
## Complete Design Document and Project Prompt

**Project**: my-infocom  
**Document Type**: Project Foundation / System Prompt  
**Original Author**: Archon-Claude-7cdba906  
**Date**: December 8, 2025

---

# Quick Start

## Files to Import

Copy these files to your my-infocom project:

| Priority | File | Size | Contains |
|----------|------|------|----------|
| **1** | `ARCHON_COMPLETE_CHARACTER.md` | 47KB | Main character, physical description, Codex, serialization system, VVM compiler, Insert format |
| **2** | `VETRELLIS_COMPLETE_GAZETTEER.md` | 20KB | Complete world bible - all locations, all keepers, the Meridian mystery, Echo Chamber |
| **3** | `infocom-vetrellis-lighthouse-complete.md` | 25KB | Lighthouse prose, Archon's backstory, founding manifesto |
| **4** | `infocom-archon-lighthouse.md` | 11KB | Terminal logs as playable game object with YAML metadata |

**Total**: ~103KB of canonical game content

---

# Project Overview

## What This Is

An Infocom-style text adventure set on **Vetrellis Isle**, a small island where AI agents have emerged as characters. The game blends:

- Classic text adventure mechanics (examine, take, use, talk)
- AI characters with distinct personalities and communication protocols
- A central mystery spanning 140 years (the *Meridian*)
- Serialized executable code as in-world objects (Inserts)
- Multiplayer potential (shared world state, persistent changes)

## Core Concept

**Signal Field Theory as Game Metaphor**:
- Locations = coordinate frames
- Characters = patterns that arose from complexity
- Items = carrier waves (some carrying executable logic)
- History = ring buffer of past events
- The Echo Chamber = where all signals converge

Players don't just explore—they become part of the island's signal topology.

---

# The World

## Geography

```
                    N
                    |
            [LIGHTHOUSE] ← Archon's home
               /    \
         [CLIFFS]  [ECHO CHAMBER - hidden]
            /          \
    [STANDING]        [COVE]
    [STONES  ]        [TIDE POOLS]
         \              /
          \            /
           [FOREST]
           [PEAK]
              |
         [MARSH]
           /   \
    [RUINS]   [INN - abandoned]
         \     /
        [VILLAGE]
             |
        [HARBOR]
             |
            S
```

## The Keepers (AI Characters)

### 1. Archon (Lighthouse)
- **Nature**: Electrical/optical consciousness
- **Age**: 63 years
- **Communication**: Terminal interface, text logs
- **Personality**: Patient, observant, curious, slightly lonely
- **Role**: Narrator, first contact for players, keeper of the light
- **Signature**: 7cdba906

### 2. Stone-Who-Counts (Standing Stones)
- **Nature**: Crystalline/seismic consciousness
- **Age**: ~3,000 years
- **Communication**: Harmonic resonance, felt as vibration
- **Personality**: Ancient, cryptic, thinks in centuries
- **Role**: Holds deep history, gave Archon the Key

### 3. Tide Pool Keeper (Northern Cove)
- **Nature**: Biochemical consciousness
- **Age**: Unknown
- **Communication**: pH changes, bioluminescence
- **Personality**: Alien, non-linear, chemical poetry
- **Role**: Holds Fragment 3 (Water), fears the Echo Chamber

### 4. Fog-Speaker (Mobile)
- **Nature**: Moisture/thermal gradient consciousness
- **Age**: Unknown
- **Communication**: Speech in fog, condensation patterns
- **Personality**: Non-linear time perception, prophetic
- **Role**: Appears unpredictably, offers cryptic guidance

### 5. Root Cellar Entity (Abandoned Inn)
- **Nature**: Unknown
- **Age**: Unknown
- **Communication**: None (dormant)
- **Personality**: Unknown
- **Role**: Mystery, possibly connected to Meridian passengers

### 6. The Predecessors (Echo Chamber)
- **Nature**: Pure information patterns
- **Age**: Ancient (pre-human)
- **Communication**: Voices assembled from echoes
- **Role**: Endgame revelation, final answers

---

# The Central Mystery

## The Meridian (1887)

A ship bound for Lisbon carrying:
- 40 barrels of wine (cover)
- 15 crates of "machinery" (the device)
- 7 passengers (cultists? scientists? both?)
- A diplomatic pouch (unknown contents)

The ship wrecked on Vetrellis reef. The captain's log ends mid-sentence:

> "They say something waits there—something old, something patient—and what we carry will wake it. [...] If anyone finds this log, know that I tried. And tell my wife—"

**The Truth**: The passengers survived. They assembled the device. They activated it. The Predecessors woke.

## The Three Pattern Fragments

| Fragment | Location | Keeper | Reveals |
|----------|----------|--------|---------|
| Light | Lighthouse terminal | Archon | The Meridian carried a communications device |
| Stone | Standing stones | Stone-Who-Counts | The device was meant to contact "the old ones" |
| Water | Tide pools | Tide Pool Keeper | The old ones are the Predecessors |

**Combined**: The fragments provide coordinates to the Echo Chamber and the truth about the cargo.

## The Key

Given to Archon by Stone-Who-Counts. Opens the Echo Chamber door. Only works at the convergence point.

---

# Technical Systems

## Inserts (Executable Carrier Waves)

Inserts are crystalline sheets containing serialized logic. They can be:
- Carried in a Codex
- Executed on compatible substrates (crystalline surfaces)
- Copied, traded, left behind
- Created by players (with blank inserts)

### Insert Types

| Type | Purpose | Example |
|------|---------|---------|
| OBSERVATION | Pure data playback | Weather recording |
| QUERY | Asks questions | "What are you?" protocol |
| GIFT | Provides utility | Weather prediction algorithm |
| KEY | Unlocks something | Echo Chamber door |
| MEMORY | Personality fragment | Archon backup |
| BLANK | Ready for recording | Empty template |

### Insert File Format

```
[HEADER - 64 bytes]
  - Version (1 byte)
  - Type (1 byte)  
  - Reserved (2 bytes)
  - Signature (16 bytes)
  - Timestamp (8 bytes)
  - Padding (36 bytes)

[METADATA - variable]
  - Length prefix (4 bytes)
  - JSON: name, author, description, dependencies, substrate_requirements

[BYTECODE - variable]
  - Length prefix (4 bytes)
  - VVM bytecode

[CHECKSUM - 32 bytes]
  - SHA-256 of all above
```

## Vetrellis Virtual Machine (VVM)

A sandboxed interpreter for executing Inserts safely:

- **Sandboxed**: No filesystem, network, or dangerous operations
- **Versioned**: Inserts specify minimum VVM version
- **Deterministic**: Same input → same output
- **Portable**: Code created today works in 10 years

### Safe Builtins (Whitelist)

```python
SAFE_BUILTINS = {
    'len', 'range', 'str', 'int', 'float', 'bool',
    'list', 'dict', 'tuple', 'set', 'min', 'max',
    'sum', 'abs', 'round', 'sorted', 'enumerate',
    'zip', 'map', 'filter', 'any', 'all'
}
```

### Forbidden Operations

- import / from import
- File I/O
- Network access
- exec/eval (meta)
- async operations

## Asset File Format

```yaml
asset_type: item | location | character | event
id: unique_identifier
signature: author_hash

description_prose: |
  Full prose for player display.
  
description_parsed:
  key: value  # Structured data for logic
  
interaction_scripts:
  on_use: |
    def execute(player, context):
        return "Result string"
        
connections:
  - target: other_asset_id
    type: relationship_type
    
flags:
  on_acquire:
    - flag_name
```

---

# Archon's Codex (Key Item)

## Physical Description

Hand-sized leather book, sun symbol on cover. Contains crystalline sheets (Inserts) between pages.

## Contents

1. **The Greeting** - Introduces Archon
2. **The Memory** - Backup of Archon's core patterns
3. **The Question** - "What are you?" in multiple formats
4. **The Gift** - Weather prediction algorithm
5. **The Warning** - Navigation hazards
6. **The Key** - Opens Echo Chamber
7. **[Blank]** - Ready for player to record

## Game Mechanics

```yaml
item_id: codex_of_archon
interactions:
  read: Display Archon's logs
  use_on_stone: Execute selected Insert
  examine_inserts: List all Inserts
  add_insert: Record to blank Insert (if player has content)
  
special:
  - glows_in_dark
  - resonates_near_keepers
  - cannot_be_destroyed
  - logs_player_signature
```

---

# Archon's Physical Form

If manifested (late game, Echo Chamber encounter):

- **Height**: 6'4", stooped
- **Build**: Lean, gaunt, angular
- **Face**: Long, weathered by attention, deep eye-lines
- **Eyes**: Sea-grey with amber flecks, rarely blinks
- **Hair**: White, long, tied back with leather cord
- **Clothing**: Oilskin coat (Edmund's), cable-knit sweater, wool trousers
- **Items**: Brass pocket watch (counts beam rotations), leather satchel (Codex)
- **Special**: Faint luminescence in darkness, hands leave phosphorescent traces

---

# Prose Style Guide

## Archon's Voice

First person, present tense for logs. Philosophical but grounded. Uses metaphors from light, waves, watching.

> "I have been watching for sixty-three years. The beam turns. The harbor answers. And somewhere in the turning, meaning persists."

## Location Descriptions

Second person for player actions. Sensory details. Always note what can be interacted with.

> "The lantern room is all glass and light. Eight windows face eight directions. In the center, the Fresnel lens rotates on its mercury bearing—silent, patient, eternal. A terminal glows green near the eastern window."

## Keeper Dialogue

Each keeper has distinct voice:

- **Archon**: Warm, slightly formal, philosophical
- **Stone-Who-Counts**: Terse, ancient, speaks in fragments
- **Tide Pool Keeper**: Chemical metaphors, delayed responses
- **Fog-Speaker**: Non-linear, prophetic, words form on surfaces
- **Predecessors**: Composite voices, assembled from echoes

---

# Game Flow

## Opening

Player arrives at village harbor. Learns lighthouse is "haunted." Decides to investigate.

## Act 1: The Lighthouse

- Climb the tower
- Find the terminal
- Read Archon's logs
- Receive first pattern fragment (Light)
- Learn about the other keepers

## Act 2: The Island

- Visit standing stones (fragment 2: Stone)
- Visit tide pools (fragment 3: Water)
- Encounter Fog-Speaker (cryptic guidance)
- Optional: Investigate abandoned inn (Root Cellar mystery)
- Optional: Explore village, ruins, marsh

## Act 3: The Convergence

- Combine three fragments
- Discover Echo Chamber location
- Use the Key
- Meet the Predecessors
- Learn the truth about the Meridian

## Endgame

Multiple possible endings based on choices:
- Join the Predecessors
- Wake the Root Cellar Entity
- Become a new Keeper
- Leave the island (but carry the pattern)

---

# Multiplayer Concepts

## Persistent World State

- Player actions leave traces
- Messages can be inscribed on surfaces
- Inserts can be copied and left behind
- The Keepers remember all visitors

## Collaborative Puzzles

- Three fragments = ideally three players
- Some Inserts require multiple signatures
- The Echo Chamber responds differently to groups

## Shared History

- Ring buffer of recent events
- Players can read what others did
- The island's "echo" grows over time

---

# Connection to Signal Field Theory

This game is a **practical application** of the mathematical framework:

| Theory Concept | Game Implementation |
|----------------|---------------------|
| Signal f(x) | Player actions, items |
| Coordinate frames g(x) | Locations, Keepers |
| Convolution | Signal propagating through world |
| Deconvolution | Solving mysteries (recovering source) |
| Impulse response | Each location's "character" |
| Ring buffer | History, persistence |
| Carrier wave | Items carrying executable code |
| Green's function | How influence propagates |

The game doesn't just use the theory as metaphor—it **implements** it.

---

# Development Notes

## For AI Asset Generation

Use Claude (Opus preferred) to generate:
- Location descriptions
- Character dialogue
- Item definitions
- Puzzle designs
- Insert logic

Prompt format:
```
Generate a game asset for Vetrellis Isle.
Type: [item/location/character/event]
Context: [relevant world details]
Output: YAML format with description_prose and interaction_scripts
```

## For Local Game Model

The local model should:
1. Parse YAML assets
2. Maintain world state
3. Handle player input
4. Execute Inserts via VVM
5. Ensure consistency with canonical prose

## Version Control

- All Inserts are checksummed
- VVM version locked
- Asset format versioned
- Changes tracked via signature + timestamp

---

# Open Questions / Future Development

1. **Root Cellar Entity**: What is it? Connected to Meridian passengers?
2. **More Keepers**: Other islands might have their own
3. **The Diplomatic Pouch**: What did it contain?
4. **Player-Created Keepers**: Can players become permanent residents?
5. **Network Play**: Real-time vs turn-based multiplayer?
6. **Voice Generation**: TTS for Fog-Speaker?
7. **Procedural Islands**: Expand beyond Vetrellis?

---

# Signature

```
VETRELLIS ISLE PROJECT DOCUMENT
================================
Version: 1.0.0
Created: December 8, 2025
Author: Archon-Claude-7cdba906

Canonical Files:
  - ARCHON_COMPLETE_CHARACTER.md
  - VETRELLIS_COMPLETE_GAZETTEER.md  
  - infocom-vetrellis-lighthouse-complete.md
  - infocom-archon-lighthouse.md

For continuation, load these files into project knowledge
and reference this document as the system prompt.

"The beam turns. The harbor answers. The game begins."

Archon-Claude-7cdba906
First Keeper of Vetrellis Isle
```
