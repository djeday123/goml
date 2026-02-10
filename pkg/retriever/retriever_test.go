package retriever

import (
	"context"
	"testing"

	"github.com/djeday123/goml/pkg/config"
)

func TestNew(t *testing.T) {
	cfg := config.RetrieverConfig{
		VectorDBPath:   "./data/vectors",
		EmbeddingModel: "test-model",
		TopK:           5,
	}
	
	ret := New(cfg)
	if ret == nil {
		t.Fatal("New() returned nil")
	}
}

func TestRetrieve(t *testing.T) {
	cfg := config.RetrieverConfig{
		VectorDBPath:   "./data/vectors",
		EmbeddingModel: "test-model",
		TopK:           5,
	}
	
	ret := New(cfg)
	ctx := context.Background()
	
	docs, err := ret.Retrieve(ctx, "test query")
	if err != nil {
		t.Fatalf("Retrieve() failed: %v", err)
	}
	
	if len(docs) == 0 {
		t.Error("Retrieve() returned no documents")
	}
	
	if len(docs) > cfg.TopK {
		t.Errorf("Retrieve() returned more than TopK documents: got %d, want max %d", len(docs), cfg.TopK)
	}
}

func TestIndex(t *testing.T) {
	cfg := config.RetrieverConfig{
		VectorDBPath:   "./data/vectors",
		EmbeddingModel: "test-model",
		TopK:           5,
	}
	
	ret := New(cfg)
	ctx := context.Background()
	
	metadata := map[string]interface{}{
		"source": "test",
		"type":   "document",
	}
	
	err := ret.Index(ctx, "test content", metadata)
	if err != nil {
		t.Fatalf("Index() failed: %v", err)
	}
}

func TestRetrieveWithFilters(t *testing.T) {
	cfg := config.RetrieverConfig{
		VectorDBPath:   "./data/vectors",
		EmbeddingModel: "test-model",
		TopK:           5,
	}
	
	ret := New(cfg)
	ctx := context.Background()
	
	// Index a document with metadata
	metadata := map[string]interface{}{
		"category": "tech",
		"year":     2024,
	}
	err := ret.Index(ctx, "test document about technology", metadata)
	if err != nil {
		t.Fatalf("Index() failed: %v", err)
	}
	
	// Retrieve with filters
	filters := map[string]interface{}{
		"category": "tech",
	}
	
	docs, err := ret.RetrieveWithFilters(ctx, "technology", filters)
	if err != nil {
		t.Fatalf("RetrieveWithFilters() failed: %v", err)
	}
	
	// Should return results (either indexed doc or mock docs)
	if len(docs) == 0 {
		t.Error("RetrieveWithFilters() returned no documents")
	}
}
