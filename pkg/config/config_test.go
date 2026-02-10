package config

import "testing"

func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig()
	
	if cfg == nil {
		t.Fatal("DefaultConfig() returned nil")
	}
	
	// Test Generator config
	if cfg.Generator.ModelSize != "7B" {
		t.Errorf("Expected Generator ModelSize to be 7B, got %s", cfg.Generator.ModelSize)
	}
	
	if cfg.Generator.Temperature <= 0 {
		t.Error("Expected Generator Temperature to be positive")
	}
	
	if cfg.Generator.MaxTokens <= 0 {
		t.Error("Expected Generator MaxTokens to be positive")
	}
	
	// Test Reviewer config
	if cfg.Reviewer.ModelSize == "" {
		t.Error("Expected Reviewer ModelSize to be set")
	}
	
	if cfg.Reviewer.CompressionRatio <= 0 || cfg.Reviewer.CompressionRatio >= 1 {
		t.Errorf("Expected Reviewer CompressionRatio to be between 0 and 1, got %f", cfg.Reviewer.CompressionRatio)
	}
	
	if cfg.Reviewer.ContextWindowSize <= 0 {
		t.Error("Expected Reviewer ContextWindowSize to be positive")
	}
	
	// Test Retriever config
	if cfg.Retriever.TopK <= 0 {
		t.Error("Expected Retriever TopK to be positive")
	}
	
	if cfg.Retriever.VectorDBPath == "" {
		t.Error("Expected Retriever VectorDBPath to be set")
	}
}
