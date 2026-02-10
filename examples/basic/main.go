package main

import (
	"context"
	"fmt"
	"log"

	"github.com/djeday123/goml/pkg/config"
	"github.com/djeday123/goml/pkg/pipeline"
)

// This example demonstrates the basic usage of the specialized model system
func main() {
	// Create a default configuration
	cfg := config.DefaultConfig()
	
	// Customize if needed
	cfg.Generator.Temperature = 0.8
	cfg.Reviewer.CompressionRatio = 0.4
	cfg.Retriever.TopK = 3
	
	// Initialize the pipeline
	p := pipeline.New(cfg)
	
	ctx := context.Background()
	
	// Index some documents in the knowledge base
	documents := []struct {
		content  string
		metadata map[string]interface{}
	}{
		{
			"Go is a statically typed, compiled programming language designed at Google.",
			map[string]interface{}{"topic": "programming", "language": "go"},
		},
		{
			"The Go programming language is known for its simplicity and efficiency.",
			map[string]interface{}{"topic": "programming", "language": "go"},
		},
		{
			"Machine learning models can be deployed in various environments including cloud and edge devices.",
			map[string]interface{}{"topic": "ml", "category": "deployment"},
		},
	}
	
	for _, doc := range documents {
		if err := p.IndexDocument(ctx, doc.content, doc.metadata); err != nil {
			log.Fatalf("Failed to index document: %v", err)
		}
	}
	
	fmt.Println("=== Example 1: Simple Query ===")
	response, err := p.ProcessRequest(ctx, "What is Go?")
	if err != nil {
		log.Fatalf("Failed to process request: %v", err)
	}
	fmt.Println(response)
	
	fmt.Println("\n=== Example 2: Streaming Response ===")
	err = p.ProcessRequestStream(ctx, "Tell me about machine learning", func(chunk string) error {
		fmt.Print(chunk)
		return nil
	})
	if err != nil {
		log.Fatalf("Failed to stream response: %v", err)
	}
	fmt.Println()
	
	fmt.Println("\n=== Example 3: Conversation Summary ===")
	summary, err := p.GetConversationSummary(ctx)
	if err != nil {
		log.Fatalf("Failed to get summary: %v", err)
	}
	fmt.Println(summary)
	
	fmt.Println("\n=== Example 4: Reset Conversation ===")
	p.ResetConversation()
	fmt.Println("Conversation history cleared.")
}
