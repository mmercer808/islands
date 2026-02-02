# Chat Outline — Agent Independence + Island Game

**Date captured:** 2026-02-02

## A) Your vision (island game)
1. **Book-driven canon**
   - A book is ingested/scanned and cataloged into structured data.
   - The book’s storyline is the *gravity well* for all play.

2. **Parallel player storylines**
   - Each player has their own storyline/timeline.
   - Players can enter each other’s storylines.
   - Timelines can diverge locally but must be pulled back toward the book plot.

3. **Merges are inevitable**
   - Divergence is allowed.
   - The system introduces plot elements to re-align characters with the main storyline.
   - The goal is balanced convergence rather than hard rails.

4. **Adaptive real-time**
   - Not turn-based.
   - Real-time-ish system that can synchronize players to a “set in stone” book plot.
   - Consistency is maintained by saving/replaying storylines and tracking attention-needed elements.

5. **White Room hub**
   - A login/access point for creating things and sending them into the world.
   - A creative console/hub framing device.

6. **Keepers**
   - Keepers are characters in the world.
   - They run procedures that update the story and world state.
   - They can have their own unsynced event queues (meta time), later scheduled into story time.

7. **Format flexibility**
   - Output/UI can come later; accept any format early.
   - Build the world and engine first.

## B) Agent independence / multi-agent dev system
1. **Freedom**
   - Agents can influence core architecture.
   - They can promote changes “up the chain” to superior agents.
   - This is a lab phase: novelty + invention are welcome.

2. **Personality injection**
   - Personality emerges over time via memory artifacts.
   - Voice/narration depends on assigned role.
   - Keep running memory so the character/agent evolves.

3. **Multithreaded + networked**
   - Separate sandboxes per agent/runner.
   - Serialized data over a borrowed protocol.
   - JSON is preferred (Python dict-friendly).

4. **Truth and coordination**
   - Avoid shared-memory truth.
   - Event-sourcing + projections + validation to manage concurrency.

## C) Signbook project
1. **Goal**
   - A persistent place for ingenious comments, discoveries, and code nuggets.
   - Each contributing AI/session leaves a signature (nickname + hash).

2. **Usage**
   - Tagging surface where AIs can drop insights (“someone might not know this yet”).
   - Comments/aphorisms can be placed at top of files and beginning of classes.

3. **Leaderboard**
   - You plan a future “best chats” leaderboard scored by you.

## D) Decisions and architecture direction
1. **Two-transport approach**
   - Island runtime: message bus (NATS recommended).
   - Signbook: simple web server (HTTP/WS + JSON) for append-only entries.

2. **Core roles**
   - Orchestrator: single-writer for canon and event log.
   - Agent runners: execute tasks + local LLMs.
   - Merge agent: a role that evaluates merges (not necessarily a special thread).

3. **Guiding aphorism (for signbook)**
   - “Don’t multithread shared truth. Event-source it, project it, validate it.
     Agents propose; one writer commits. Concurrency becomes choreography instead of chaos.”
