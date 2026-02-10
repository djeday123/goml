package pipeline

import (
	"context"
	"fmt"
	"strings"

	"github.com/djeday123/goml/pkg/config"
	"github.com/djeday123/goml/pkg/generator"
	"github.com/djeday123/goml/pkg/retriever"
	"github.com/djeday123/goml/pkg/reviewer"
)

// Pipeline orchestrates the specialized model system
// It coordinates Generator, Reviewer, and Retriever components
type Pipeline struct {
	generator generator.Generator
	reviewer  reviewer.Reviewer
	retriever retriever.Retriever
	
	conversation reviewer.ConversationContext
}

// New creates a new Pipeline instance
func New(cfg *config.Config) *Pipeline {
	return &Pipeline{
		generator: generator.New(cfg.Generator),
		reviewer:  reviewer.New(cfg.Reviewer),
		retriever: retriever.New(cfg.Retriever),
		conversation: reviewer.ConversationContext{
			Messages: make([]reviewer.Message, 0),
		},
	}
}

// ProcessRequest handles a user request through the entire pipeline
func (p *Pipeline) ProcessRequest(ctx context.Context, userInput string) (string, error) {
	if ctx.Err() != nil {
		return "", ctx.Err()
	}
	
	var result strings.Builder
	result.WriteString("=== Pipeline Processing ===\n\n")
	
	// Step 1: Retrieve relevant context using RAG
	result.WriteString("Step 1: Retrieval (RAG)\n")
	docs, err := p.retriever.Retrieve(ctx, userInput)
	if err != nil {
		return "", fmt.Errorf("retrieval failed: %w", err)
	}
	
	result.WriteString(fmt.Sprintf("Retrieved %d documents\n", len(docs)))
	for i, doc := range docs {
		result.WriteString(fmt.Sprintf("  Doc %d (score: %.2f): %s\n", i+1, doc.Score, doc.Content))
	}
	result.WriteString("\n")
	
	// Step 2: Check if context compression is needed
	p.conversation.Messages = append(p.conversation.Messages, reviewer.Message{
		Role:    "user",
		Content: userInput,
	})
	
	if p.reviewer.ShouldCompress(p.conversation) {
		result.WriteString("Step 2: Compression (Context too large, compressing...)\n")
		compressed, err := p.reviewer.Compress(ctx, p.conversation)
		if err != nil {
			return "", fmt.Errorf("compression failed: %w", err)
		}
		result.WriteString(compressed + "\n\n")
	} else {
		result.WriteString("Step 2: Compression (Context size OK, skipping compression)\n\n")
	}
	
	// Step 3: Build enhanced prompt with retrieved context
	var enhancedPrompt strings.Builder
	enhancedPrompt.WriteString("Context from knowledge base:\n")
	for _, doc := range docs {
		enhancedPrompt.WriteString(fmt.Sprintf("- %s\n", doc.Content))
	}
	enhancedPrompt.WriteString("\nUser query: ")
	enhancedPrompt.WriteString(userInput)
	
	// Step 4: Generate response
	result.WriteString("Step 3: Generation\n")
	response, err := p.generator.Generate(ctx, enhancedPrompt.String())
	if err != nil {
		return "", fmt.Errorf("generation failed: %w", err)
	}
	result.WriteString(response + "\n\n")
	
	// Step 5: Add assistant response to conversation
	p.conversation.Messages = append(p.conversation.Messages, reviewer.Message{
		Role:    "assistant",
		Content: response,
	})
	
	result.WriteString("=== Pipeline Complete ===\n")
	
	return result.String(), nil
}

// ProcessRequestStream handles a request with streaming generation
func (p *Pipeline) ProcessRequestStream(ctx context.Context, userInput string, callback func(string) error) error {
	if ctx.Err() != nil {
		return ctx.Err()
	}
	
	// Step 1: Retrieve
	callback("=== Retrieving context ===\n")
	docs, err := p.retriever.Retrieve(ctx, userInput)
	if err != nil {
		return fmt.Errorf("retrieval failed: %w", err)
	}
	
	callback(fmt.Sprintf("Retrieved %d documents\n\n", len(docs)))
	
	// Step 2: Check compression
	p.conversation.Messages = append(p.conversation.Messages, reviewer.Message{
		Role:    "user",
		Content: userInput,
	})
	
	if p.reviewer.ShouldCompress(p.conversation) {
		callback("=== Compressing context ===\n")
		compressed, err := p.reviewer.Compress(ctx, p.conversation)
		if err != nil {
			return fmt.Errorf("compression failed: %w", err)
		}
		callback(compressed + "\n\n")
	}
	
	// Step 3: Generate with streaming
	callback("=== Generating response ===\n")
	
	var enhancedPrompt strings.Builder
	enhancedPrompt.WriteString("Context: ")
	for _, doc := range docs {
		enhancedPrompt.WriteString(doc.Content + " ")
	}
	enhancedPrompt.WriteString("\nQuery: " + userInput)
	
	err = p.generator.GenerateStream(ctx, enhancedPrompt.String(), callback)
	if err != nil {
		return fmt.Errorf("generation failed: %w", err)
	}
	
	return nil
}

// GetConversationSummary returns a distilled summary of the conversation
func (p *Pipeline) GetConversationSummary(ctx context.Context) (string, error) {
	return p.reviewer.Distill(ctx, p.conversation)
}

// IndexDocument adds a document to the knowledge base
func (p *Pipeline) IndexDocument(ctx context.Context, content string, metadata map[string]interface{}) error {
	return p.retriever.Index(ctx, content, metadata)
}

// ResetConversation clears the conversation history
func (p *Pipeline) ResetConversation() {
	p.conversation.Messages = make([]reviewer.Message, 0)
}
