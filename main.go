package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"time"

	_ "github.com/vugar/goml/backend/cpu"
	"github.com/vugar/goml/backend"
	"github.com/vugar/goml/nn"
	"github.com/vugar/goml/ops"
	"github.com/vugar/goml/optim"
	"github.com/vugar/goml/tensor"
	"github.com/vugar/goml/tokenizer"
)

func main() {
	fmt.Println("=== GoML — LLM Training from Scratch ===\n")

	// Load data
	data, err := os.ReadFile("data/shakespeare.txt")
	if err != nil {
		fmt.Println("No data file, using inline text")
		data = []byte(inlineText)
	}

	tok := tokenizer.NewByteTokenizer()
	allTokens := tok.Encode(string(data))
	
	// 90/10 split
	splitIdx := int(float64(len(allTokens)) * 0.9)
	trainTokens := allTokens[:splitIdx]
	evalTokens := allTokens[splitIdx:]
	
	fmt.Printf("Data: %d tokens (train: %d, eval: %d)\n", len(allTokens), len(trainTokens), len(evalTokens))

	// Model
	cfg := nn.TinyConfig()
	cfg.MaxSeqLen = 64
	
	model, _ := nn.NewLLM(cfg, backend.CPU0)
	totalParams := model.CountParameters()
	fmt.Printf("Model: %d params (%.2f K)\n", totalParams, float64(totalParams)/1e3)

	// Optimizer
	opt := optim.NewAdamW(model.Parameters(), 3e-4)

	// Training config
	seqLen := 32
	batchSize := 1
	steps := 3000
	logEvery := 200
	evalEvery := 1000
	genEvery := 1000

	fmt.Printf("Config: batch=%d, seqLen=%d, lr=3e-4, steps=%d\n\n", batchSize, seqLen, steps)

	totalStart := time.Now()
	smoothLoss := float64(0)
	bestEval := math.MaxFloat64

	for step := 1; step <= steps; step++ {
		stepStart := time.Now()

		// Cosine LR schedule with warmup
		lr := optim.CosineSchedule(step, 200, steps, 3e-4, 3e-5)
		opt.SetLR(lr)

		// Get batch
		inputs, targets := getBatch(trainTokens, batchSize, seqLen)

		// Forward
		logits, cache, err := model.ForwardWithCache(inputs)
		if err != nil {
			fmt.Printf("step %d: forward err: %v\n", step, err)
			continue
		}

		// Check for NaN
		if hasNaN(logits.ToFloat32Slice()) {
			fmt.Printf("step %d: NaN in logits\n", step)
			continue
		}

		// Loss
		loss, _ := ops.CrossEntropyLoss(logits, targets)
		lossVal := float64(loss.ToFloat32Slice()[0])

		if math.IsNaN(lossVal) || math.IsInf(lossVal, 0) {
			fmt.Printf("step %d: bad loss %.4f\n", step, lossVal)
			continue
		}

		if smoothLoss == 0 {
			smoothLoss = lossVal
		} else {
			smoothLoss = 0.99*smoothLoss + 0.01*lossVal
		}

		// Backward
		opt.ZeroGrad()
		dLogits, _ := ops.CrossEntropyBackward(logits, targets)
		err = model.Backward(cache, dLogits)
		if err != nil {
			fmt.Printf("step %d: backward err: %v\n", step, err)
			continue
		}

		// Optimizer step
		opt.Step()

		elapsed := time.Since(stepStart)
		tokSec := float64(batchSize*seqLen) / elapsed.Seconds()

		if step%logEvery == 0 || step == 1 {
			fmt.Printf("step %4d | loss %.4f (smooth %.4f) | lr %.1e | %.0f tok/s | %v\n",
				step, lossVal, smoothLoss, lr, tokSec, elapsed)
		}

		if step%evalEvery == 0 {
			evalLoss := evaluate(model, evalTokens, seqLen)
			tag := ""
			if evalLoss < bestEval {
				bestEval = evalLoss
				tag = " *best"
			}
			fmt.Printf("         → eval loss: %.4f%s\n", evalLoss, tag)
		}

		if step%genEvery == 0 {
			sample := generate(model, tok, cfg, "The ", 100, 0.8)
			fmt.Printf("         → sample: %q\n", truncate(sample, 120))
		}
	}

	totalTime := time.Since(totalStart)
	fmt.Printf("\nTraining complete in %v\n", totalTime)
	fmt.Printf("Best eval loss: %.4f (random baseline: %.4f)\n", bestEval, math.Log(float64(cfg.VocabSize)))

	// Final generation
	fmt.Println("\n--- Final Generation ---")
	prompts := []string{"KING ", "To be ", "The ", "What "}
	for _, p := range prompts {
		sample := generate(model, tok, cfg, p, 200, 0.7)
		fmt.Printf("\n%q → %s\n", p, truncate(sample, 200))
	}
}

func getBatch(tokens []int64, batchSize, seqLen int) (*tensor.Tensor, *tensor.Tensor) {
	maxStart := len(tokens) - seqLen - 1
	inputData := make([]int64, batchSize*seqLen)
	targetData := make([]int64, batchSize*seqLen)

	for b := 0; b < batchSize; b++ {
		start := rand.Intn(maxStart)
		for s := 0; s < seqLen; s++ {
			inputData[b*seqLen+s] = tokens[start+s]
			targetData[b*seqLen+s] = tokens[start+s+1]
		}
	}

	inputs, _ := tensor.FromSlice(inputData, tensor.Shape{batchSize, seqLen})
	targets, _ := tensor.FromSlice(targetData, tensor.Shape{batchSize, seqLen})
	return inputs, targets
}

func evaluate(model *nn.LLM, tokens []int64, seqLen int) float64 {
	totalLoss := float64(0)
	numBatches := 10
	for i := 0; i < numBatches; i++ {
		inputs, targets := getBatch(tokens, 1, seqLen)
		logits, _ := model.Forward(inputs)
		loss, _ := ops.CrossEntropyLoss(logits, targets)
		totalLoss += float64(loss.ToFloat32Slice()[0])
	}
	return totalLoss / float64(numBatches)
}

func generate(model *nn.LLM, tok *tokenizer.ByteTokenizer, cfg nn.ModelConfig, prompt string, maxLen int, temperature float32) string {
	tokens := tok.Encode(prompt)
	for i := 0; i < maxLen; i++ {
		start := 0
		if len(tokens) > cfg.MaxSeqLen {
			start = len(tokens) - cfg.MaxSeqLen
		}
		window := tokens[start:]
		input, _ := tensor.FromSlice(window, tensor.Shape{1, len(window)})
		logits, _ := model.Forward(input)
		data := logits.ToFloat32Slice()
		lastOff := (len(window) - 1) * cfg.VocabSize
		last, _ := tensor.FromSlice(data[lastOff:lastOff+cfg.VocabSize], tensor.Shape{cfg.VocabSize})
		next := int64(nn.TopKSample(last, 40, temperature))
		tokens = append(tokens, next)
	}
	return tok.Decode(tokens)
}

func hasNaN(data []float32) bool {
	for _, v := range data {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return true
		}
	}
	return false
}

func truncate(s string, maxLen int) string {
	if len(s) > maxLen {
		return s[:maxLen] + "..."
	}
	return s
}

const inlineText = `KING HENRY:
Once more unto the breach, dear friends, once more;
Or close the wall up with our English dead.
In peace there's nothing so becomes a man
As modest stillness and humility.`
