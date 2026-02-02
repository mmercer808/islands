# Island Game v1 — Foundation for Alternate Islands

This document proposes a *minimal playable core* that supports:
- book-driven canon
- parallel player timelines
- keeper-driven convergence (plot gravity)
- alternate islands (distinct settings sharing engine rules)

## 1) Core loop (real-time, event-sourced)
Players produce **actions** continuously:
- speak, move, inspect, craft, trade, rest, travel
- meta actions: “enter other timeline”, “review missed plot points”

The engine produces:
- narration
- consequences
- keeper interventions (scheduled into story time)
- updates to projections (canon + timelines)

Everything is recorded as events.

## 2) Minimal world model (engine-agnostic)
### Entities
- Character (player/NPC/keeper)
- Location (rooms, zones, island regions)
- Item (objects, keys, artifacts)
- Thread (timeline branch)
- Scene (a slice of story time tied to a location + cast)

### Links
- Character located_at Location
- Location connected_to Location
- Character knows Character (relationship edges)
- Scene references Entities

## 3) Two-time system
### Story time
What the player experiences: pacing, scenes, “now”.

### Meta time (keeper time)
Background reasoning:
- check plot beat distance
- check contradictions
- choose interventions
- queue proposals

Orchestrator schedules keeper proposals into story time so it feels fair.

## 4) Plot gravity (how convergence stays fun)
Plot gravity is *not* teleporting players to the plot.
It is adding **meaningful constraints** and **tempting affordances**:

Hook archetypes (starter set):
1. **Omen**: foreshadowing that changes player priorities
2. **Summons**: an NPC request with time pressure
3. **Revelation**: an item or memory that recontextualizes choices
4. **Tradeoff**: a resource gate that creates a natural path
5. **Rumor net**: social information that points toward canon beats

Each hook is an event, not a rewrite.

## 5) Alternate islands (the expansion mechanism)
An “island” is a package:
- canon seed (book catalog + initial canon events)
- island rules (validators, physics/economy knobs)
- keeper roster (voices + specialties)
- content libraries (locations, characters, items)

Engine stays the same.
Only packages change.

## 6) Minimal playable milestone
To call v1 “alive”, we need:
- 1 island package (tiny: 3 locations, 5 NPCs, 10 items)
- 1 keeper NPC running plot gravity
- 1 player timeline with divergence detection
- a text interface (CLI is fine) producing events + narration

Once that works, we can scale sideways: more islands, more keepers, more agents.
