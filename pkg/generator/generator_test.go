package generator

import (
	"context"
	"strings"
	"testing"

	"github.com/djeday123/goml/pkg/config"
)

func TestNew(t *testing.T) {
	cfg := config.GeneratorConfig{
		ModelSize:   "7B",
		Temperature: 0.7,
		MaxTokens:   2048,
		Endpoint:    "http://localhost:8080",
	}
	
	gen := New(cfg)
	if gen == nil {
		t.Fatal("New() returned nil")
	}
}

func TestGenerate(t *testing.T) {
	cfg := config.GeneratorConfig{
		ModelSize:   "7B",
		Temperature: 0.7,
		MaxTokens:   2048,
	}
	
	gen := New(cfg)
	ctx := context.Background()
	
	result, err := gen.Generate(ctx, "test prompt")
	if err != nil {
		t.Fatalf("Generate() failed: %v", err)
	}
	
	if result == "" {
		t.Error("Generate() returned empty result")
	}
	
	// Should contain model size
	if !strings.Contains(result, "7B") {
		t.Error("Result should contain model size")
	}
}

func TestGenerateWithCancelledContext(t *testing.T) {
	cfg := config.GeneratorConfig{
		ModelSize:   "7B",
		Temperature: 0.7,
		MaxTokens:   2048,
	}
	
	gen := New(cfg)
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately
	
	_, err := gen.Generate(ctx, "test prompt")
	if err == nil {
		t.Error("Expected error with cancelled context")
	}
}

func TestGenerateStream(t *testing.T) {
	cfg := config.GeneratorConfig{
		ModelSize:   "7B",
		Temperature: 0.7,
		MaxTokens:   2048,
	}
	
	gen := New(cfg)
	ctx := context.Background()
	
	chunks := make([]string, 0)
	callback := func(chunk string) error {
		chunks = append(chunks, chunk)
		return nil
	}
	
	err := gen.GenerateStream(ctx, "test prompt", callback)
	if err != nil {
		t.Fatalf("GenerateStream() failed: %v", err)
	}
	
	if len(chunks) == 0 {
		t.Error("GenerateStream() produced no chunks")
	}
}
