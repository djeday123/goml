package reviewer

import (
	"context"
	"fmt"
	"strings"

	"github.com/djeday123/goml/pkg/config"
)

// ConversationContext represents the conversation history
type ConversationContext struct {
	Messages []Message
}

// Message represents a single message in the conversation
type Message struct {
	Role    string // "user" or "assistant"
	Content string
}

// Reviewer is a tiny model (1-3B) that compresses and distills conversation context
type Reviewer interface {
	// Compress reduces the conversation context while preserving key information
	Compress(ctx context.Context, conversation ConversationContext) (string, error)
	
	// Distill creates a concise summary of the conversation
	Distill(ctx context.Context, conversation ConversationContext) (string, error)
	
	// ShouldCompress determines if the context needs compression
	ShouldCompress(conversation ConversationContext) bool
}

// reviewer implements the Reviewer interface
type reviewer struct {
	config config.ReviewerConfig
}

// New creates a new Reviewer instance
func New(cfg config.ReviewerConfig) Reviewer {
	return &reviewer{
		config: cfg,
	}
}

// Compress reduces the conversation context while preserving key information
func (r *reviewer) Compress(ctx context.Context, conversation ConversationContext) (string, error) {
	if ctx.Err() != nil {
		return "", ctx.Err()
	}
	
	// Calculate total tokens (simplified: word count)
	totalTokens := 0
	for _, msg := range conversation.Messages {
		totalTokens += len(strings.Fields(msg.Content))
	}
	
	// Apply compression
	targetTokens := int(float64(totalTokens) * r.config.CompressionRatio)
	
	result := fmt.Sprintf("[Reviewer %s] Compressed %d messages (%d tokens -> %d tokens)\n",
		r.config.ModelSize, len(conversation.Messages), totalTokens, targetTokens)
	
	// In a real implementation, this would use the model to compress
	result += "Compressed context: Key points preserved with reduced verbosity."
	
	return result, nil
}

// Distill creates a concise summary of the conversation
func (r *reviewer) Distill(ctx context.Context, conversation ConversationContext) (string, error) {
	if ctx.Err() != nil {
		return "", ctx.Err()
	}
	
	// Build the distillation input
	var conversationText strings.Builder
	for _, msg := range conversation.Messages {
		conversationText.WriteString(fmt.Sprintf("%s: %s\n", msg.Role, msg.Content))
	}
	
	result := fmt.Sprintf("[Reviewer %s] %s\n\n",
		r.config.ModelSize, r.config.DistillationPrompt)
	
	// In a real implementation, this would use the model to distill
	result += fmt.Sprintf("Summary: Conversation with %d messages covering various topics.",
		len(conversation.Messages))
	
	return result, nil
}

// ShouldCompress determines if the context needs compression
func (r *reviewer) ShouldCompress(conversation ConversationContext) bool {
	// Calculate total tokens (simplified: word count)
	totalTokens := 0
	for _, msg := range conversation.Messages {
		totalTokens += len(strings.Fields(msg.Content))
	}
	
	// Compress if we're approaching the context window size
	threshold := int(float64(r.config.ContextWindowSize) * 0.8)
	return totalTokens > threshold
}
