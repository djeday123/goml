# goml - Compound LLM System

A specialized model system that uses three coordinated components instead of one massive model, optimizing for efficiency and performance.

## Architecture

Instead of burning through tokens with a single large model, **goml** uses a pipeline of specialized models:

### 🚀 Generator (7B Parameters)
- **Purpose**: Fast, efficient text generation
- **Model Size**: ~7B parameters, heavily optimized
- **Features**: 
  - Standard and streaming generation
  - Configurable temperature and max tokens
  - Low latency responses

### 🔄 Reviewer (1-3B Parameters)  
- **Purpose**: Context compression and conversation distillation
- **Model Size**: 1-3B parameters for efficiency
- **Features**:
  - Continuous context compression
  - Intelligent conversation summarization
  - Automatic compression triggers
  - Configurable compression ratios

### 🔍 Retriever (RAG Module)
- **Purpose**: Real-time knowledge retrieval
- **Features**:
  - Vector-based document search
  - External source integration
  - Embedding-based similarity matching
  - Metadata filtering

## Pipeline Flow

```
User Input
    ↓
1. Retriever → Fetch relevant context from knowledge base
    ↓
2. Reviewer → Check & compress conversation context if needed
    ↓
3. Generator → Generate response with enhanced context
    ↓
Output
```

## Installation

```bash
# Clone the repository
git clone https://github.com/djeday123/goml.git
cd goml

# Build the CLI
go build -o goml ./cmd/goml

# Run
./goml
```

## Quick Start

```go
package main

import (
    "context"
    "fmt"
    "github.com/djeday123/goml/pkg/config"
    "github.com/djeday123/goml/pkg/pipeline"
)

func main() {
    // Initialize with default config
    cfg := config.DefaultConfig()
    
    // Create pipeline
    p := pipeline.New(cfg)
    
    // Index some documents
    ctx := context.Background()
    p.IndexDocument(ctx, "Your knowledge base content here", map[string]interface{}{
        "source": "manual",
    })
    
    // Process a request
    response, err := p.ProcessRequest(ctx, "Your query here")
    if err != nil {
        panic(err)
    }
    
    fmt.Println(response)
}
```

## Configuration

Customize each component:

```go
cfg := &config.Config{
    Generator: config.GeneratorConfig{
        ModelSize:   "7B",
        Temperature: 0.7,
        MaxTokens:   2048,
        Endpoint:    "http://localhost:8080",
    },
    Reviewer: config.ReviewerConfig{
        ModelSize:          "3B",
        CompressionRatio:   0.3,  // Compress to 30% of original
        ContextWindowSize:  4096,
        DistillationPrompt: "Summarize key points:",
        Endpoint:           "http://localhost:8081",
    },
    Retriever: config.RetrieverConfig{
        VectorDBPath:    "./data/vectors",
        EmbeddingModel:  "sentence-transformers/all-MiniLM-L6-v2",
        TopK:            5,  // Return top 5 matches
        ExternalSources: []string{"https://api.example.com"},
        Endpoint:        "http://localhost:8082",
    },
}
```

## CLI Usage

The CLI provides an interactive interface:

```bash
$ ./goml

> What is machine learning?
# Pipeline processes query through RAG → Reviewer → Generator

> summary
# Get conversation summary

> reset  
# Clear conversation history

> exit
# Quit
```

## API Reference

### Pipeline

```go
// Create new pipeline
p := pipeline.New(cfg)

// Process request (blocking)
response, err := p.ProcessRequest(ctx, userInput)

// Process with streaming
err := p.ProcessRequestStream(ctx, userInput, func(chunk string) error {
    fmt.Print(chunk)
    return nil
})

// Get conversation summary
summary, err := p.GetConversationSummary(ctx)

// Index documents
err := p.IndexDocument(ctx, content, metadata)

// Reset conversation
p.ResetConversation()
```

### Generator

```go
gen := generator.New(cfg.Generator)

// Generate text
response, err := gen.Generate(ctx, prompt)

// Stream generation
err := gen.GenerateStream(ctx, prompt, func(chunk string) error {
    fmt.Print(chunk)
    return nil
})
```

### Reviewer

```go
rev := reviewer.New(cfg.Reviewer)

// Compress context
compressed, err := rev.Compress(ctx, conversation)

// Distill summary
summary, err := rev.Distill(ctx, conversation)

// Check if compression needed
needsCompression := rev.ShouldCompress(conversation)
```

### Retriever

```go
ret := retriever.New(cfg.Retriever)

// Retrieve documents
docs, err := ret.Retrieve(ctx, query)

// Retrieve with filters
docs, err := ret.RetrieveWithFilters(ctx, query, filters)

// Index document
err := ret.Index(ctx, content, metadata)
```

## Benefits

1. **Token Efficiency**: Smaller specialized models use fewer tokens than large general models
2. **Lower Latency**: Fast 7B generator provides quick responses
3. **Context Management**: Automatic compression prevents context overflow
4. **Enhanced Accuracy**: RAG provides relevant, up-to-date information
5. **Scalability**: Each component can be scaled independently
6. **Cost Effective**: Smaller models reduce API costs and infrastructure requirements

## Use Cases

- **Customer Support**: RAG for knowledge base + fast generation
- **Code Assistants**: Context compression for long coding sessions
- **Research Tools**: Real-time retrieval from large document collections
- **Chatbots**: Efficient conversation management with context distillation

## License

MIT

## Contributing

Contributions welcome! Please open an issue or submit a PR.
