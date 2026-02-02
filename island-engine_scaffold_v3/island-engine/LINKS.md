# Links and Pointers

## Repo bundles
- v2 zip: sandbox:/mnt/data/island-engine_scaffold_v2.zip
- v3 zip (this build): sandbox:/mnt/data/island-engine_scaffold_v3.zip

## Key docs
- docs/design.md
- docs/chat_outline.md
- docs/island_game_v1.md
- docs/design_appendix_white_room.md
- docs/current_files_index.md
- TODO.md

## Run signbook server (with a face)
```bash
pip install -r requirements.txt
uvicorn services.signbook_server.main:app --host 127.0.0.1 --port 8088
```
Open `/` in your browser.
