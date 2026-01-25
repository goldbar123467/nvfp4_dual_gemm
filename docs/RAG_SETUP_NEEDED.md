# RAG Brain Setup Required

## Status
The RAG brain (PostgreSQL + pgvector with Ollama embeddings) is not available.

## Error
```
Memory rejected: Failed to generate embedding: Ollama connection error: All connection attempts failed
```

## What's Needed

### 1. Start Ollama Service
```bash
# Check if Ollama is installed
which ollama

# Start Ollama
ollama serve

# Or if using systemd
sudo systemctl start ollama
```

### 2. Verify Embedding Model
```bash
# Pull the embedding model if needed
ollama pull nomic-embed-text

# Or whatever model the RAG brain uses
ollama list
```

### 3. Test RAG Brain Connection
```bash
# From Claude Code, use:
# mcp__rag-brain__recall with query "test"
```

## Ready Content

Once RAG brain is available, run the Archivist agent again OR manually import from:
- `/home/ubuntu/projects/Shark-Tank-for-GPUMODE.COM/ARCHIVIST_MANIFEST.md`

This manifest contains 27 memories ready for storage:
- 8 Round Outcomes
- 6 Technical Insights
- 4 Anti-Patterns
- 4 Success Patterns
- 4 Key Decisions
- 1 Performance Baseline

## Quick Re-Run Command

Resume the Archivist agent to retry storage once Ollama is running.
