package retriever

import (
	"context"
	"fmt"
	"strings"

	"github.com/djeday123/goml/pkg/config"
)

// Document represents a retrieved document
type Document struct {
	Content  string
	Source   string
	Score    float64
	Metadata map[string]interface{}
}

// Retriever is a RAG module that searches external sources in real-time
type Retriever interface {
	// Retrieve finds relevant documents based on the query
	Retrieve(ctx context.Context, query string) ([]Document, error)
	
	// RetrieveWithFilters finds documents with additional filtering
	RetrieveWithFilters(ctx context.Context, query string, filters map[string]interface{}) ([]Document, error)
	
	// Index adds a document to the retrieval system
	Index(ctx context.Context, content string, metadata map[string]interface{}) error
}

// retriever implements the Retriever interface
type retriever struct {
	config config.RetrieverConfig
	// In a real implementation, this would hold vector DB connections
	documents []Document
}

// New creates a new Retriever instance
func New(cfg config.RetrieverConfig) Retriever {
	return &retriever{
		config:    cfg,
		documents: make([]Document, 0),
	}
}

// Retrieve finds relevant documents based on the query
func (r *retriever) Retrieve(ctx context.Context, query string) ([]Document, error) {
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}
	
	// In a real implementation, this would:
	// 1. Generate embeddings for the query using the embedding model
	// 2. Search the vector database for similar documents
	// 3. Return top-k results
	
	// Mock implementation: simulate retrieval
	results := make([]Document, 0)
	
	// Simulate finding relevant documents
	for i := 0; i < r.config.TopK && i < len(r.documents); i++ {
		doc := r.documents[i]
		// Simple keyword matching for mock
		if strings.Contains(strings.ToLower(doc.Content), strings.ToLower(query)) {
			results = append(results, doc)
		}
	}
	
	// If no matches, create mock results
	if len(results) == 0 {
		for i := 0; i < min(r.config.TopK, 3); i++ {
			results = append(results, Document{
				Content: fmt.Sprintf("Retrieved document %d related to: %s", i+1, query),
				Source:  fmt.Sprintf("source_%d", i+1),
				Score:   0.9 - float64(i)*0.1,
				Metadata: map[string]interface{}{
					"embedding_model": r.config.EmbeddingModel,
					"index":           i,
				},
			})
		}
	}
	
	return results, nil
}

// RetrieveWithFilters finds documents with additional filtering
func (r *retriever) RetrieveWithFilters(ctx context.Context, query string, filters map[string]interface{}) ([]Document, error) {
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}
	
	// Get base results
	results, err := r.Retrieve(ctx, query)
	if err != nil {
		return nil, err
	}
	
	// Apply filters
	filtered := make([]Document, 0)
	for _, doc := range results {
		match := true
		for key, value := range filters {
			if docValue, ok := doc.Metadata[key]; !ok || docValue != value {
				match = false
				break
			}
		}
		if match {
			filtered = append(filtered, doc)
		}
	}
	
	return filtered, nil
}

// Index adds a document to the retrieval system
func (r *retriever) Index(ctx context.Context, content string, metadata map[string]interface{}) error {
	if ctx.Err() != nil {
		return ctx.Err()
	}
	
	// In a real implementation, this would:
	// 1. Generate embeddings for the content
	// 2. Store in vector database
	// 3. Index for fast retrieval
	
	// Mock implementation: just store in memory
	doc := Document{
		Content:  content,
		Source:   "indexed",
		Score:    1.0,
		Metadata: metadata,
	}
	
	r.documents = append(r.documents, doc)
	
	return nil
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
