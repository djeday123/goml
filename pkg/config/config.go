package config

// Config holds the configuration for the entire specialized model system
type Config struct {
	Generator GeneratorConfig `json:"generator"`
	Reviewer  ReviewerConfig  `json:"reviewer"`
	Retriever RetrieverConfig `json:"retriever"`
}

// GeneratorConfig configures the small, fast generation model
type GeneratorConfig struct {
	ModelSize   string  `json:"model_size"`   // e.g., "7B"
	Temperature float64 `json:"temperature"`
	MaxTokens   int     `json:"max_tokens"`
	Endpoint    string  `json:"endpoint"` // API endpoint or model path
}

// ReviewerConfig configures the context compression model
type ReviewerConfig struct {
	ModelSize          string  `json:"model_size"` // e.g., "1B", "3B"
	CompressionRatio   float64 `json:"compression_ratio"`
	ContextWindowSize  int     `json:"context_window_size"`
	DistillationPrompt string  `json:"distillation_prompt"`
	Endpoint           string  `json:"endpoint"`
}

// RetrieverConfig configures the RAG retrieval module
type RetrieverConfig struct {
	VectorDBPath    string   `json:"vector_db_path"`
	EmbeddingModel  string   `json:"embedding_model"`
	TopK            int      `json:"top_k"`
	ExternalSources []string `json:"external_sources"`
	Endpoint        string   `json:"endpoint"`
}

// DefaultConfig returns a default configuration
func DefaultConfig() *Config {
	return &Config{
		Generator: GeneratorConfig{
			ModelSize:   "7B",
			Temperature: 0.7,
			MaxTokens:   2048,
			Endpoint:    "http://localhost:8080",
		},
		Reviewer: ReviewerConfig{
			ModelSize:          "3B",
			CompressionRatio:   0.3,
			ContextWindowSize:  4096,
			DistillationPrompt: "Summarize the following conversation, keeping key points and context:",
			Endpoint:           "http://localhost:8081",
		},
		Retriever: RetrieverConfig{
			VectorDBPath:    "./data/vectors",
			EmbeddingModel:  "sentence-transformers/all-MiniLM-L6-v2",
			TopK:            5,
			ExternalSources: []string{},
			Endpoint:        "http://localhost:8082",
		},
	}
}
