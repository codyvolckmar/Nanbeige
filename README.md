I open-sourced my one-command setup for running Nanbeige 4.3 33B as a private coding agent on a single rented GPU 🔥

https://github.com/codyvolckmar/Nanbeige

The goal: a self-hosted agent that can read a codebase, call tools, run commands and build a website end to end, on hardware I rent by the hour. On one RTX A6000 (48 GB), driven by the pi coding agent, it rebuilt an entire website using about 22% of its 256K context window.

Why Nanbeige
Independent Artificial Analysis numbers back it up: Intelligence Index 20, top of the ≤40B open-weights class (tied with Qwen3.8 27B), and 29.8% on Terminal-Bench 4 vs Qwen3.8 27B's 5.6%. Text-only, Apache-2.0, built for agentic coding.

The catch: context memory
Nanbeige keeps a context cache in all 52 layers, so 256K tokens needs ~27.6 GB of cache even at 8-bit. On 48 GB that means Q3_K_M weights + an 8-bit cache ≈ 45 GB, with every layer on the GPU. One setting trades it for 128K context with near-lossless Q6_K weights.

Speed
38.7 tok/s on an empty context, ~15 tok/s at ~58K tokens. Every new token re-reads 15 GB of weights plus the growing cache, so at long context, memory bandwidth is the bottleneck, not compute.

What the script does
• Picks the longest context that fits first, then the best weights. If it doesn't fit, it fails loudly instead of silently offloading to the CPU
• llama.cpp with the model's own template: native XML tool calls, plus preserve_thinking for agent work
• Prompt cache in host RAM, so a quick side request doesn't evict a 200K-token conversation
• Lossless n-gram speculative decoding for code rewrites (benchmarking next)
• Resumable, verified download with a disk-space check
• Port locked to the IP you SSH in from: a firewall rule, no VPN, and one flag to re-lock
• A tool-call smoke test before you connect an agent

What the research showed
• Qwen3.8 27B is the strongest alternative. Only 16 of its 64 layers keep a cache, so it fits 256K with 8-bit weights on the same card and slows down far less as context grows
• Flint (40B total, 2.5B active) runs at ~146 tok/s, but its benchmarks are still vendor-only
• Sub-20B models don't hold up for agentic work. The best scores 5 on AA's index vs 20 at 27–33B. They're fine for bulk document extraction. Want speed? Lower the thinking effort on a 27B instead

Next
Measure the n-gram speedup on real edits, A/B Nanbeige vs Qwen3.8 27B on the same site with a frontier model as the final reviewer, and test exllamav3's EXL3 format, which should fit 5-bit weights at the full 256K on the same card.

Open models doing real agent work on one rented GPU aren't a toy anymore. If you try it, I'd love to hear what you build.

#AI #LLM #LocalLLM #llamacpp #OpenSource #AIAgents #MachineLearning #SelfHosted #OpenWeights #AICoding #BuildInPublic
