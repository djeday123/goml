# Examples

This directory contains example applications demonstrating how to use the goml specialized model system.

## Basic Example

The basic example demonstrates core functionality:

```bash
cd examples/basic
go run main.go
```

This example shows:
- Configuration setup
- Document indexing
- Simple queries
- Streaming responses
- Conversation summarization
- Context reset

## Running Examples

From the repository root:

```bash
# Run basic example
go run ./examples/basic

# Or build and run
go build -o basic-example ./examples/basic
./basic-example
```

## What Each Example Demonstrates

### Basic (`examples/basic`)
- Creating and configuring the pipeline
- Indexing documents with metadata
- Processing queries through the full pipeline
- Using streaming mode
- Getting conversation summaries
- Resetting conversation state
