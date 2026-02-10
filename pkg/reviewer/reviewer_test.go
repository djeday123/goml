package reviewer

import (
	"context"
	"strings"
	"testing"

	"github.com/djeday123/goml/pkg/config"
)

func TestNew(t *testing.T) {
	cfg := config.ReviewerConfig{
		ModelSize:          "3B",
		CompressionRatio:   0.3,
		ContextWindowSize:  4096,
		DistillationPrompt: "Summarize:",
	}
	
	rev := New(cfg)
	if rev == nil {
		t.Fatal("New() returned nil")
	}
}

func TestCompress(t *testing.T) {
	cfg := config.ReviewerConfig{
		ModelSize:          "3B",
		CompressionRatio:   0.3,
		ContextWindowSize:  4096,
		DistillationPrompt: "Summarize:",
	}
	
	rev := New(cfg)
	ctx := context.Background()
	
	conversation := ConversationContext{
		Messages: []Message{
			{Role: "user", Content: "Hello, how are you?"},
			{Role: "assistant", Content: "I'm doing well, thank you!"},
		},
	}
	
	result, err := rev.Compress(ctx, conversation)
	if err != nil {
		t.Fatalf("Compress() failed: %v", err)
	}
	
	if result == "" {
		t.Error("Compress() returned empty result")
	}
}

func TestDistill(t *testing.T) {
	cfg := config.ReviewerConfig{
		ModelSize:          "3B",
		CompressionRatio:   0.3,
		ContextWindowSize:  4096,
		DistillationPrompt: "Summarize:",
	}
	
	rev := New(cfg)
	ctx := context.Background()
	
	conversation := ConversationContext{
		Messages: []Message{
			{Role: "user", Content: "What is AI?"},
			{Role: "assistant", Content: "AI is artificial intelligence."},
			{Role: "user", Content: "Tell me more."},
			{Role: "assistant", Content: "AI involves machine learning and deep learning."},
		},
	}
	
	result, err := rev.Distill(ctx, conversation)
	if err != nil {
		t.Fatalf("Distill() failed: %v", err)
	}
	
	if result == "" {
		t.Error("Distill() returned empty result")
	}
}

func TestShouldCompress(t *testing.T) {
	cfg := config.ReviewerConfig{
		ModelSize:          "3B",
		CompressionRatio:   0.3,
		ContextWindowSize:  100, // Small window for testing
		DistillationPrompt: "Summarize:",
	}
	
	rev := New(cfg)
	
	// Small conversation - should not compress
	smallConversation := ConversationContext{
		Messages: []Message{
			{Role: "user", Content: "Hi"},
			{Role: "assistant", Content: "Hello"},
		},
	}
	
	if rev.ShouldCompress(smallConversation) {
		t.Error("Should not compress small conversation")
	}
	
	// Large conversation - should compress
	largeConversation := ConversationContext{
		Messages: []Message{
			{Role: "user", Content: strings.Repeat("word ", 50)},
			{Role: "assistant", Content: strings.Repeat("word ", 50)},
		},
	}
	
	if !rev.ShouldCompress(largeConversation) {
		t.Error("Should compress large conversation")
	}
}
