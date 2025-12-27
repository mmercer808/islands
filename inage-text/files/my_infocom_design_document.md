# my-infocom Entity & Prose System Design

## Purpose of This Document

This document explains **how** and **why** the system works—not just what it is. It's meant to be handed off to another project, so every major decision includes the reasoning behind it.

---

## Part 1: The Core Problem

### What We're Building

A game where **locations, objects, and characters** come from real-world images (Street View, photos), get processed through vision models, and become **playable, narratively rich game content**.

The challenge: images give us *raw detection* ("there's a building, a car, a person"), but games need *prose, relationships, and interaction*.

### Why Infocom-Style Parsing Is Outdated

The classic `> GET LAMP` parser has problems:
1. **Guess-the-verb frustration** — players don't know what words work
2. **Brittle vocabulary** — "take", "get", "grab", "pick up" should all work
3. **No context awareness** — the parser doesn't know what's relevant right now
4. **Static prose** — descriptions don't adapt to what you've seen or done

**Our approach**: The entity graph stores *structured data*, and prose is **assembled on demand** from fragments. The interaction layer can be parser-based, choice-based, or hybrid—the underlying system supports all of them.

---

## Part 2: System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         GAME RUNTIME                                │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐  │
│  │ Interaction │───▶│   Query     │───▶│   Prose Compositor      │  │
│  │   Layer     │    │   Engine    │    │   (assembles output)    │  │
│  └─────────────┘    └─────────────┘    └─────────────────────────┘  │
│         │                 │                        │                │
│         ▼                 ▼                        ▼                │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                      ENTITY GRAPH                               ││
│  │  Nodes (entities) + Links (relationships) + State               ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
                                 ▲
                                 │ (import)
┌─────────────────────────────────────────────────────────────────────┐
│                     IMAGE PIPELINE (offline)                        │
│  Detection ──▶ VLM Description ──▶ LLM Narrative ──▶ Entity Export  │
└─────────────────────────────────────────────────────────────────────┘
```

### The Three Major Jobs

| Job | When It Happens | What It Does |
|-----|-----------------|--------------|
| **1. Entity Creation** | Offline/authoring | Images → structured entities with prose fragments |
| **2. Graph Assembly** | Game load | Entities registered, links established, indexes built |
| **3. Runtime Query & Prose** | During play | Player actions → graph queries → assembled prose output |

---

## Part 3: Prose Fragmentation (The Key Insight)

### Why Fragments, Not Monolithic Descriptions

In classic IF, a room has one description string. Problems:
- Can't adapt to state ("the door is now open")
- Can't hide/reveal details dynamically
- Can't vary for replay value
- Can't integrate procedural content (from image pipeline)

**Solution**: Prose is stored as **tagged fragments** that get **composed at runtime**.

### Fragment Types

```
┌─────────────────────────────────────────────────────────────────┐
│                        PROSE FRAGMENT                           │
├─────────────────────────────────────────────────────────────────┤
│ text: "A brass key lies beneath the desk."                      │
│ category: ITEM_PRESENCE                                         │
│ priority: 50                                                    │
│ conditions: [entity.revealed = true, first_visit = true]        │
│ tags: [key, discovery, study]                                   │
│ variations: ["You spot a brass key under the desk.",            │
│              "Something metallic glints beneath the desk."]     │
└─────────────────────────────────────────────────────────────────┘
```

### Fragment Categories (What Kinds of Prose Exist)

| Category | Purpose | Example |
|----------|---------|---------|
| `BASE_DESCRIPTION` | Core room/object description | "A wood-paneled study." |
| `ATMOSPHERIC` | Mood, lighting, weather | "Dust motes drift in pale light." |
| `ITEM_PRESENCE` | Noting items are here | "A letter sits on the desk." |
| `STATE_CHANGE` | Something has changed | "The drawer now hangs open." |
| `SENSORY` | Sight, sound, smell, touch | "The smell of old tobacco lingers." |
| `HISTORY` | Backstory hints | "Faded photographs line the mantle." |
| `NPC_AMBIENT` | Character presence/activity | "Jenkins stands by the door, waiting." |
| `DISCOVERY` | First-time revelations | "You notice a hidden compartment!" |

### How Composition Works (The Process)

**Step 1: Gather Applicable Fragments**
```
Player enters Study
  → Get all fragments linked to Study entity
  → Filter by conditions (is item revealed? first visit? state?)
  → Each fragment either passes or fails its conditions
```

**Step 2: Sort by Priority and Category**
```
Ordering (high to low):
  1. BASE_DESCRIPTION (always first)
  2. ATMOSPHERIC (set the mood)
  3. STATE_CHANGE (what's different)
  4. ITEM_PRESENCE (what's here)
  5. NPC_AMBIENT (who's here)
  6. SENSORY (immersion details)
  7. DISCOVERY (new information last)
```

**Step 3: Assemble and Vary**
```
For each fragment:
  - If it has variations, pick one (random or contextual)
  - Apply text transforms (pronoun resolution, tense)
  - Add to output buffer
  
Join fragments into paragraphs with appropriate spacing
```

**Step 4: Output**
```
Final prose sent to player, composed from 3-10 fragments
depending on context richness and verbosity setting
```

### Example: Composing a Room Description

**Situation**: Player enters Study for the first time. Desk hasn't been searched yet. Jenkins is in the Foyer (not here).

**Fragments that qualify**:
1. `[BASE]` "Oak-paneled walls lined with bookshelves..."
2. `[ATMOSPHERIC]` "Dust motes drift in the still air."
3. `[ITEM_PRESENCE]` "Papers are scattered across a massive desk."
4. `[SENSORY]` "The smell of old leather and pipe tobacco lingers."

**Fragments that DON'T qualify**:
- `[DISCOVERY]` "A brass key lies beneath the desk." — condition `desk.searched` is false
- `[NPC_AMBIENT]` "Jenkins stands by the door." — Jenkins not in this location

**Composed Output**:
> Oak-paneled walls lined with bookshelves. A massive desk dominates the room. Dust motes drift in the still air, and the smell of old leather and pipe tobacco lingers. Papers are scattered across the desk's surface.

---

## Part 4: The Entity Graph (Data Structure)

### Why a Graph?

Games have complex relationships that aren't hierarchical:
- A key can be IN a drawer, which is PART_OF a desk, which is IN a room
- The key UNLOCKS a door in a DIFFERENT room
- A character KNOWS about the key but WON'T_TELL you unless trust is high

Trees can't model this. A graph with typed edges can.

### Node (Abstract Base)

**Responsibility**: Be a connection point in the graph.

**What it tracks**:
- Unique ID
- Incoming links (other nodes pointing to this one)
- Outgoing links (this node pointing to others)
- Tags (for fast categorical queries)
- Observer list (for state change notifications)

**Why abstract**: Not everything in the graph is a full "entity." We might have waypoints, abstract concepts, or narrative beats that connect but aren't examinable objects.

### Entity (Concrete Node)

**Responsibility**: Represent a game-world thing with prose and interaction.

**What it adds beyond Node**:
- Name and type (LOCATION, ITEM, CHARACTER, etc.)
- Prose fragments (the text content)
- Vocabulary (nouns and adjectives for reference resolution)
- State (hidden, locked, open, etc.)
- Properties bag (arbitrary key-value data)
- Interaction handlers (what happens when you DO things to it)

### Link (Edge)

**Responsibility**: Connect two nodes with a typed, conditional relationship.

**Three categories** (as you specified):

**RELATIONAL** — Describes spatial/physical/social relationships
- These are mostly static (the desk is IN the study)
- Bidirectional links auto-create inverses (NORTH_OF ↔ SOUTH_OF)
- Used for: navigation, containment, possession, NPC relationships

**LOGICAL** — Describes game-logic dependencies
- Often conditional (active only when predicate is true)
- Used for: locks/keys, puzzle dependencies, revelation triggers
- Examples: key UNLOCKS door, searching desk REVEALS key

**WILDCARD** — Custom/narrative relationships
- The kind string is arbitrary (you define it)
- Used for: thematic connections, clue networks, narrative beats
- Examples: letter CLUE_CONNECTS_TO key (both have "J.W."), photograph REMINDS_OF character

### Why Conditional Links Matter

A static graph can't model: "The key becomes visible after you search the desk."

With conditional links:
```
Link: desk --[REVEALS]--> key
Condition: desk.properties.searched == true
```

When composing prose, the system checks: is the REVEALS link active? If not, the key's ITEM_PRESENCE fragment doesn't get included.

---

## Part 5: The Query Engine (Finding Things)

### The Problem

Player types "examine the old letter" or clicks on [Letter]. The system needs to:
1. Figure out what "old letter" refers to (reference resolution)
2. Find that entity in the current context
3. Get all relevant information for the response

### Query Process

**Step 1: Reference Resolution**
```
Input: "old letter"
Tokenize: ["old", "letter"]
Classify: "letter" is likely a noun, "old" is likely an adjective

For each entity in current scope:
  - Does it have "letter" in its nouns? 
  - Does it have "old" in its adjectives?
  - Return matches with confidence scores
```

**Step 2: Scope Determination**
```
What's "in scope" for the player right now?
  - Current location
  - Items in current location (via CONTAINS links)
  - Items in player inventory
  - Visible NPCs
  - Items inside open containers

NOT in scope:
  - Items in other rooms
  - Items inside closed containers
  - Hidden items (until revealed)
```

**Step 3: Disambiguation**
```
If multiple matches:
  - Rank by specificity (more adjective matches = better)
  - Rank by proximity (in hand > on ground > across room)
  - If still ambiguous, ask player: "Which do you mean: the torn letter or the sealed letter?"
```

### Query Builder (Fluent API)

For internal/authoring queries (not player input):
```
Find all takeable items in the study that haven't been examined:

query = (NodeQuery()
         .with_type(ITEM)
         .with_tag('takeable')
         .connected_to(study)
         .where(lambda e: not e.properties.examined))
```

This is useful for:
- Game logic ("are there any clues left unfound?")
- Hints system ("suggest something the player missed")
- Testing ("verify all items are reachable")

---

## Part 6: Interaction Layer (Multiple Paradigms)

### Why Decouple Interaction from Graph

The entity graph and prose system don't assume HOW the player interacts. This lets us support:

| Style | How It Works | Pros/Cons |
|-------|--------------|-----------|
| **Parser** | Player types commands | Classic IF feel; steep learning curve |
| **Choice** | Player picks from options | Accessible; can feel limiting |
| **Hybrid** | Choices + occasional typing | Best of both; complex to implement |
| **Point-and-click** | GUI with clickable objects | Visual; loses text atmosphere |

### The Interaction Contract

Whatever the UI, it produces an **Action**:
```
Action {
  verb: "examine" | "take" | "talk" | "use" | ...
  target: EntityReference (resolved or unresolved)
  instrument: EntityReference? (for "use X on Y")
  context: {current_location, inventory, game_state}
}
```

The **Action Processor** then:
1. Resolves references to actual entities
2. Checks preconditions (can you do this action to this entity?)
3. Invokes the entity's handler OR a default handler
4. Collects state changes
5. Triggers prose composition for the response

### Entity Handlers

Entities can register handlers for specific actions:
```
desk.on('search', handler) 
  → runs when player searches the desk
  → handler returns prose and causes state changes
  → those state changes may activate conditional links
```

Default handlers exist for common actions:
- `examine` → return entity's description (composed from fragments)
- `take` → if tagged 'takeable', move to inventory
- `drop` → move from inventory to current location

---

## Part 7: Image Pipeline Integration

### What the Pipeline Produces

The image processing pipeline (from our architecture doc) outputs:
```
{
  "name": "Victorian Townhouse",
  "entity_type": "structure",
  "detection": {
    "label": "building",
    "confidence": 0.94,
    "bbox": {...}
  },
  "vlm_description": {
    "summary": "A three-story Victorian townhouse with bay windows",
    "mood": "mysterious",
    "architectural_style": "Victorian Gothic",
    "condition": "weathered"
  },
  "narrative": {
    "prose_description": "The townhouse looms against the grey sky...",
    "history_hints": ["Once belonged to a prominent family", ...],
    "rumors": ["They say no one's lived there for decades", ...],
    "sensory_details": {"sight": "Peeling paint, dark windows", ...}
  }
}
```

### The Conversion Job

**Job: `pipeline_output_to_entity`**

**Input**: Raw pipeline JSON for one detected object/scene
**Output**: An Entity with properly structured prose fragments

**Process**:
```
1. Map entity_type string to Entity.Type enum
   "structure" → LOCATION
   "vehicle" → ITEM or VEHICLE
   "person" → CHARACTER

2. Create base fragment from prose_description
   category: BASE_DESCRIPTION
   text: narrative.prose_description

3. Create sensory fragments (one per sense)
   for each key in sensory_details:
     category: SENSORY
     tags: [key]  # 'sight', 'sound', 'smell'
     text: sensory_details[key]

4. Create history fragments
   for each hint in history_hints:
     category: HISTORY
     conditions: [discovered: true] or [examine_count > 1]
     text: hint

5. Create atmospheric fragment from mood
   category: ATMOSPHERIC
   text: (generated or template-based from mood)

6. Attach all fragments to entity

7. Set tags based on detection label and attributes
   "building" → ['structure', 'enterable', 'landmark']
   
8. Store original detection data in properties
   (useful for debugging, re-processing)
```

### Batch Processing a Location

When processing a Street View scene, multiple detections occur. The job:

```
1. Process the "main" scene as a LOCATION entity
2. Process each detection as a child entity
3. Establish CONTAINS links from location to children
4. Infer additional links from spatial analysis
   - Objects near each other get NEAR links
   - Objects inside bounds of another get IN links
5. Export as entity graph subset (JSON or direct)
```

---

## Part 8: Graph Assembly & Serialization

### Why Serialization Matters

- **Authoring**: Entities created offline need to be saved
- **Game saves**: Player progress means graph state changes
- **Modularity**: Different "areas" can be separate files, loaded on demand

### Serialization Format

```json
{
  "graph_name": "Mystery Manor",
  "entities": [
    {
      "id": "study_001",
      "type": "LOCATION",
      "name": "Study",
      "fragments": [
        {"category": "BASE", "text": "...", "priority": 100},
        {"category": "ATMOSPHERIC", "text": "...", "priority": 80}
      ],
      "vocabulary": {"nouns": ["study", "room"], "adjectives": ["dark"]},
      "properties": {"visited": false},
      "tags": ["indoor", "ground_floor"]
    }
  ],
  "links": [
    {
      "source": "study_001",
      "target": "foyer_001", 
      "type": "RELATIONAL",
      "kind": "WEST_OF",
      "bidirectional": true
    },
    {
      "source": "desk_001",
      "target": "key_001",
      "type": "LOGICAL",
      "kind": "REVEALS",
      "condition": "desk_001.properties.searched == true"
    }
  ]
}
```

### Loading Job

```
1. Parse JSON
2. For each entity definition:
   a. Create Entity object
   b. Attach fragments
   c. Register with graph (builds indexes)
3. For each link definition:
   a. Resolve source/target IDs to entities
   b. Parse condition if present (into callable)
   c. Create Link object
   d. Register with graph (updates node link lists)
4. Validate graph integrity
   - All link targets exist
   - No orphan entities (unless intentional)
   - Required tags present
```

---

## Part 9: Runtime Flow (Putting It All Together)

### Example: Player Examines the Desk

```
┌─────────────────────────────────────────────────────────────────────┐
│ PLAYER INPUT: "examine desk"                                        │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ REFERENCE RESOLUTION                                                │
│ - Tokenize: ["examine", "desk"]                                     │
│ - Identify verb: "examine"                                          │
│ - Identify target: "desk"                                           │
│ - Query current scope for entities matching "desk"                  │
│ - Result: desk_001 entity                                           │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ ACTION PROCESSING                                                   │
│ - Create Action(verb="examine", target=desk_001)                    │
│ - Check: is desk_001 in scope? YES                                  │
│ - Check: does desk_001 handle "examine"? NO (use default)           │
│ - Default examine: compose description from fragments               │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ PROSE COMPOSITION                                                   │
│ - Get fragments attached to desk_001                                │
│ - Filter by conditions (all pass, desk hasn't been searched)        │
│ - Sort by category/priority                                         │
│ - Select variations if available                                    │
│ - Join into paragraph                                               │
│ - Result: "A massive oak desk dominates this corner..."             │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STATE UPDATE                                                        │
│ - Mark desk_001.properties.examined = true                          │
│ - Increment desk_001.properties.interaction_count                   │
│ - Notify observers of state change                                  │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ OUTPUT TO PLAYER                                                    │
│ "A massive oak desk dominates this corner of the study. Its         │
│  surface is cluttered with papers and old photographs. The          │
│  drawers look like they might hold more secrets."                   │
└─────────────────────────────────────────────────────────────────────┘
```

### Example: Player Searches the Desk (Triggers Revelation)

```
┌─────────────────────────────────────────────────────────────────────┐
│ PLAYER INPUT: "search desk"                                         │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ ACTION PROCESSING                                                   │
│ - desk_001 has custom handler for "search"                          │
│ - Handler runs:                                                     │
│   a. Set desk_001.properties.searched = true                        │
│   b. Return prose: "You rifle through the desk..."                  │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ LINK ACTIVATION CHECK                                               │
│ - System checks all LOGICAL links from desk_001                     │
│ - Link desk_001 --[REVEALS]--> key_001 has condition:               │
│   "desk_001.properties.searched == true"                            │
│ - Condition NOW TRUE → link activates                               │
│ - key_001 state changes: hidden → revealed                          │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ PROSE COMPOSITION (includes revelation)                             │
│ - Handler's base prose: "You rifle through the desk drawers..."     │
│ - key_001 now revealed, so its DISCOVERY fragment activates         │
│ - Append: "Beneath some papers, you find a small brass key."        │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ OUTPUT TO PLAYER                                                    │
│ "You rifle through the desk drawers, finding old receipts,          │
│  faded photographs, and dust. Beneath some papers in the            │
│  bottom drawer, you find a small brass key."                        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Part 10: Jobs Summary (What Needs to Be Built)

### Offline / Authoring Jobs

| Job | Input | Output | Notes |
|-----|-------|--------|-------|
| **Image Pipeline** | Image file | Pipeline JSON | Existing architecture |
| **Entity Conversion** | Pipeline JSON | Entity object | Map VLM output to fragments |
| **Fragment Authoring** | Manual input | Fragment objects | Tool/editor needed |
| **Graph Export** | EntityGraph | JSON file | Serialization |
| **Graph Import** | JSON file | EntityGraph | Deserialization |

### Runtime Jobs

| Job | Input | Output | Notes |
|-----|-------|--------|-------|
| **Reference Resolution** | Player words | Entity candidates | Vocabulary matching |
| **Scope Determination** | Game state | Set of accessible entities | Graph traversal |
| **Action Dispatch** | Action object | Handler result | Polymorphic dispatch |
| **Prose Composition** | Entity + context | Assembled text | Fragment filtering/sorting |
| **State Propagation** | Property change | Link activations | Condition checking |
| **Graph Query** | NodeQuery | List of entities | Index-assisted search |

---

## Part 11: What This Document Doesn't Cover (Yet)

- **NPC conversation system** — How dialogue trees integrate with the graph
- **Puzzle validation** — Ensuring puzzles are solvable
- **Procedural generation** — Creating links/relationships automatically
- **Multiplayer considerations** — Graph synchronization
- **Performance optimization** — Caching composed prose, lazy loading areas
- **Editor/tooling** — How authors create and test content

These are future design documents.

---

## Appendix: Glossary

| Term | Definition |
|------|------------|
| **Entity** | A game-world thing (location, item, character) that can be examined and interacted with |
| **Node** | Abstract graph vertex; entities are nodes, but not all nodes are entities |
| **Link** | Typed, directional edge connecting two nodes |
| **Fragment** | A piece of prose text with category, conditions, and priority |
| **Composition** | The process of assembling fragments into player-facing prose |
| **Scope** | The set of entities currently accessible/visible to the player |
| **Reference Resolution** | Mapping player words ("the old key") to a specific entity |
| **Handler** | A function that executes when an action is performed on an entity |
| **Condition** | A predicate that determines if a link/fragment is active |

---

*Document version: 1.0*
*For: my-infocom project handoff*
