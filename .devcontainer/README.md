# PGW live demo (Codespaces)

The demo starts by itself and opens in a new browser tab. The first start after creating the
codespace takes about 2 minutes to generate masks; later starts take about 1 minute.

- **Open it again:** Ports tab → "PGW live demo" (port 8080) → globe icon.
- **Private by default:** only you (signed in to GitHub) can open the link. For a client meeting,
  right-click the port → Port Visibility → Public, and set it back to Private afterwards.
- **Logs:** `/tmp/pgw-demo.log` (startup) and `~/pgw-demo/logs/` (gateway, service, factory, UI).
- **Restart:** `pkill -f client_demo/run.py; nohup bash .devcontainer/start-demo.sh > /tmp/pgw-demo.log 2>&1 &`

The codespace stops after 30 minutes without activity, which keeps it inside GitHub's free
monthly hours. A 4-core machine answers about twice as fast as a 2-core one but uses the free
hours twice as fast.
