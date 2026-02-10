package pipeline

import (
	"context"
	"strings"
	"testing"

	"github.com/djeday123/goml/pkg/config"
)

func TestNew(t *testing.T) {
	cfg := config.DefaultConfig()
	p := New(cfg)
	
	if p == nil {
		t.Fatal("New() returned nil")
	}
}

func TestProcessRequest(t *testing.T) {
	cfg := config.DefaultConfig()
	p := New(cfg)
	ctx := context.Background()
	
	result, err := p.ProcessRequest(ctx, "test query")
	if err != nil {
		t.Fatalf("ProcessRequest() failed: %v", err)
	}
	
	if result == "" {
		t.Error("ProcessRequest() returned empty result")
	}
	
	// Should contain pipeline steps
	if !strings.Contains(result, "Pipeline Processing") {
		t.Error("Result should contain pipeline processing info")
	}
}

func TestProcessRequestStream(t *testing.T) {
	cfg := config.DefaultConfig()
	p := New(cfg)
	ctx := context.Background()
	
	chunks := make([]string, 0)
	callback := func(chunk string) error {
		chunks = append(chunks, chunk)
		return nil
	}
	
	err := p.ProcessRequestStream(ctx, "test query", callback)
	if err != nil {
		t.Fatalf("ProcessRequestStream() failed: %v", err)
	}
	
	if len(chunks) == 0 {
		t.Error("ProcessRequestStream() produced no output")
	}
}

func TestGetConversationSummary(t *testing.T) {
	cfg := config.DefaultConfig()
	p := New(cfg)
	ctx := context.Background()
	
	// Add some conversation
	_, err := p.ProcessRequest(ctx, "first query")
	if err != nil {
		t.Fatalf("ProcessRequest() failed: %v", err)
	}
	
	summary, err := p.GetConversationSummary(ctx)
	if err != nil {
		t.Fatalf("GetConversationSummary() failed: %v", err)
	}
	
	if summary == "" {
		t.Error("GetConversationSummary() returned empty summary")
	}
}

func TestIndexDocument(t *testing.T) {
	cfg := config.DefaultConfig()
	p := New(cfg)
	ctx := context.Background()
	
	metadata := map[string]interface{}{
		"source": "test",
	}
	
	err := p.IndexDocument(ctx, "test document content", metadata)
	if err != nil {
		t.Fatalf("IndexDocument() failed: %v", err)
	}
}

func TestResetConversation(t *testing.T) {
	cfg := config.DefaultConfig()
	p := New(cfg)
	ctx := context.Background()
	
	// Add some conversation
	_, err := p.ProcessRequest(ctx, "test query")
	if err != nil {
		t.Fatalf("ProcessRequest() failed: %v", err)
	}
	
	// Reset
	p.ResetConversation()
	
	// Summary should reflect empty conversation
	summary, err := p.GetConversationSummary(ctx)
	if err != nil {
		t.Fatalf("GetConversationSummary() failed: %v", err)
	}
	
	// Should indicate 0 messages
	if !strings.Contains(summary, "0 messages") {
		t.Log("Summary after reset:", summary)
	}
}
