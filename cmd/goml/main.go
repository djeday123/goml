package main

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/djeday123/goml/pkg/config"
	"github.com/djeday123/goml/pkg/pipeline"
)

func main() {
	fmt.Println("╔══════════════════════════════════════════════════════════╗")
	fmt.Println("║      Specialized Model System - Compound LLM             ║")
	fmt.Println("║                                                          ║")
	fmt.Println("║  Components:                                             ║")
	fmt.Println("║    • Generator  - Fast 7B parameter model                ║")
	fmt.Println("║    • Reviewer   - Context compression (1-3B)             ║")
	fmt.Println("║    • Retriever  - Real-time RAG search                   ║")
	fmt.Println("╚══════════════════════════════════════════════════════════╝")
	fmt.Println()

	// Initialize with default configuration
	cfg := config.DefaultConfig()
	
	// Create pipeline
	p := pipeline.New(cfg)
	
	// Seed some example documents
	ctx := context.Background()
	seedKnowledgeBase(ctx, p)
	
	fmt.Println("System initialized. Type 'help' for commands, 'exit' to quit.")
	
	// Interactive loop
	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("> ")
		if !scanner.Scan() {
			break
		}
		
		input := strings.TrimSpace(scanner.Text())
		if input == "" {
			continue
		}
		
		if input == "exit" || input == "quit" {
			fmt.Println("Goodbye!")
			break
		}
		
		if input == "help" {
			printHelp()
			continue
		}
		
		if input == "summary" {
			summary, err := p.GetConversationSummary(ctx)
			if err != nil {
				fmt.Printf("Error getting summary: %v\n", err)
				continue
			}
			fmt.Println("\n" + summary)
			continue
		}
		
		if input == "reset" {
			p.ResetConversation()
			fmt.Println("Conversation reset.")
			continue
		}
		
		// Process the request
		response, err := p.ProcessRequest(ctx, input)
		if err != nil {
			fmt.Printf("Error: %v\n", err)
			continue
		}
		
		fmt.Println("\n" + response)
	}
}

func seedKnowledgeBase(ctx context.Context, p *pipeline.Pipeline) {
	documents := []struct {
		content  string
		metadata map[string]interface{}
	}{
		{
			content: "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
			metadata: map[string]interface{}{
				"topic":  "ML",
				"source": "knowledge_base",
			},
		},
		{
			content: "Neural networks are computing systems inspired by biological neural networks in animal brains.",
			metadata: map[string]interface{}{
				"topic":  "Neural Networks",
				"source": "knowledge_base",
			},
		},
		{
			content: "Large Language Models (LLMs) are trained on massive text datasets to understand and generate human-like text.",
			metadata: map[string]interface{}{
				"topic":  "LLM",
				"source": "knowledge_base",
			},
		},
		{
			content: "Retrieval-Augmented Generation (RAG) combines retrieval of relevant documents with text generation for more accurate responses.",
			metadata: map[string]interface{}{
				"topic":  "RAG",
				"source": "knowledge_base",
			},
		},
		{
			content: "Model compression techniques reduce the size of neural networks while maintaining performance, enabling deployment on resource-constrained devices.",
			metadata: map[string]interface{}{
				"topic":  "Optimization",
				"source": "knowledge_base",
			},
		},
	}
	
	for _, doc := range documents {
		if err := p.IndexDocument(ctx, doc.content, doc.metadata); err != nil {
			fmt.Printf("Warning: Failed to index document: %v\n", err)
		}
	}
	
	fmt.Println("Knowledge base seeded with example documents.")
}

func printHelp() {
	fmt.Println("\nAvailable commands:")
	fmt.Println("  <query>   - Process a query through the pipeline")
	fmt.Println("  summary   - Get a distilled summary of the conversation")
	fmt.Println("  reset     - Clear conversation history")
	fmt.Println("  help      - Show this help message")
	fmt.Println("  exit/quit - Exit the program")
	fmt.Println()
}
