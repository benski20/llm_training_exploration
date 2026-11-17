# Live demo on a Hugging Face Space

The Space image contains the PGW package and the client demo (`scripts/client_demo`). Assemble the
upload folder as: `Dockerfile`, `README.md` and `constraints.txt` from here, plus `pgw/` holding
`pyproject.toml`, `README.md`, `src/` and `scripts/client_demo/`. Create it as a private Docker
Space. Docker Spaces need a Hugging Face PRO subscription, even on free cpu-basic hardware.

At build time the image packs Qwen2.5-0.5B-Instruct (int8 weights and tokenizer). At start it
creates demo-only keys and an encrypted mask store with 300 one-time mask slots. It then runs the
compute service, gateway, wire tap and UI on port 7860, plus a mask factory daemon that refills
the store whenever it drops below 250 slots.
