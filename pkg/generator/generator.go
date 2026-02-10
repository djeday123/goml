package generator

import (
	"context"
	"fmt"

	"github.com/djeday123/goml/pkg/config"
)

// Generator is a small, fast model optimized for generating output
// Typically uses ~7B parameters for efficiency
type Generator interface {
	// Generate produces text output from the given prompt
	Generate(ctx context.Context, prompt string) (string, error)
	
	// GenerateStream produces text output as a stream
	GenerateStream(ctx context.Context, prompt string, callback func(string) error) error
}

// generator implements the Generator interface
type generator struct {
	config config.GeneratorConfig
}

// New creates a new Generator instance
func New(cfg config.GeneratorConfig) Generator {
	return &generator{
		config: cfg,
	}
}

// Generate produces text output from the given prompt
func (g *generator) Generate(ctx context.Context, prompt string) (string, error) {
	// In a real implementation, this would call an actual LLM API
	// For now, this is a mock implementation
	
	// Validate context
	if ctx.Err() != nil {
		return "", ctx.Err()
	}
	
	// Simulate generation with model info
	result := fmt.Sprintf("[Generator %s] Processing prompt: %s\n", 
		g.config.ModelSize, prompt)
	result += fmt.Sprintf("Generated response (max_tokens=%d, temp=%.2f)", 
		g.config.MaxTokens, g.config.Temperature)
	
	return result, nil
}

// GenerateStream produces text output as a stream
func (g *generator) GenerateStream(ctx context.Context, prompt string, callback func(string) error) error {
	// In a real implementation, this would stream tokens as they're generated
	// For now, simulate streaming by sending chunks
	
	if ctx.Err() != nil {
		return ctx.Err()
	}
	
	chunks := []string{
		fmt.Sprintf("[Generator %s] ", g.config.ModelSize),
		"Streaming ",
		"response ",
		"for: ",
		prompt,
	}
	
	for _, chunk := range chunks {
		if err := callback(chunk); err != nil {
			return err
		}
	}
	
	return nil
}
