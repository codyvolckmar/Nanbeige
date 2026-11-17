Open-sourced my one-command setup for running Nanbeige 4.3 33B as a private coding agent on a single rented GPU 🔥

https://github.com/codyvolckmar/Nanbeige

#LocalLLM #llamacpp #OpenSource #AIAgents #LLM #GPU #Nanbeige #SelfHosted #AICoding #OpenWeights #BuildInPublic
# Nanbeige
Nanbeige 4.3 33B llama.cpp setup and benchmark notes

1/ Spent the day turning one rented RTX A6000 (48GB) into a private coding-agent backend.

Nanbeige 4.3 33B, 256K context, served by llama.cpp, driven by the pi coding agent.

It just rebuilt an entire website using 22% of its context window. 🧵

2/ Why Nanbeige? Independent Artificial Analysis numbers back it up:

• Intelligence Index 20, top of the ≤40B class (tied with Qwen3.8 27B)
• Terminal-Bench 4: 29.8% vs Qwen3.8 27B's 5.6%

Text-only, built for agentic coding.

3/ The catch: it keeps a context cache in all 52 layers. 256K tokens = ~27.6GB of cache, even at 8-bit.

So on 48GB: Q3_K_M weights + q8_0 KV cache ≈ 45GB.

Every layer stays on the GPU. If it doesn't fit, it fails loudly instead of silently offloading to CPU.

4/ Speed reality check:

• 38.7 tok/s on an empty context
• ~15 tok/s at ~58K tokens

Every new token re-reads 15GB of weights plus the growing cache. At long context, memory bandwidth is the bottleneck, not compute.

5/ Inference tricks in the stack:

• --jinja + preserve_thinking for native XML tool calls
• Prompt cache in host RAM, so a quick side request doesn't evict a 200K-token conversation
• Lossless n-gram speculative decoding for code rewrites (benchmarking next)

6/ Locked down without a VPN:

The setup script reads the IP you SSH in from and drops everyone else at the firewall (iptables, in Docker's own DOCKER-USER chain).

Home IP changed? One flag re-locks it. No restart.

7/ Other models we dug into:

• Qwen3.8 27B: only 16 of 64 layers keep a cache → 256K with Q8 weights on 48GB, 0.82 on AA's long-context test
• Flint 40B-A2.5B: ~146 tok/s, tiny cache (vendor-only benchmarks so far)
• Ornith 1.5 35B-A3B, Swift 1.5

8/ Can sub-20B models do this? Not for agents.

The best open model under 20B scores 5 on AA's index, vs 20 at 27-33B. Granite 8B scored 0% on AA's coding test.

Small models are fine for bulk document extraction. Want speed? Lower the thinking effort on a 27B instead.

9/ Next up: measure the n-gram speedup on real edits, then A/B Nanbeige vs Qwen3.8 27B on the same site, with a frontier model as the final reviewer.

Open models running real agent work on one rented GPU is no longer a toy.
