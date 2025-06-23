package llm

import (
	"context"
	"fmt"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/discover"
)

// openvinoServer implements LlamaServer interface for NPU acceleration
type openvinoServer struct {
	port    int
	npus    discover.NpuInfoList
	options api.Options
}

// NewOpenVINOServer creates a minimal OpenVINO server implementation
// This is a placeholder that will return an error for now
func NewOpenVINOServer(npus discover.NpuInfoList, modelPath string, opts api.Options, numParallel int) (LlamaServer, error) {
	// For now, return an error to indicate OpenVINO is not yet implemented
	// This allows the scheduler to fall back to existing GPU/CPU implementations
	return nil, fmt.Errorf("OpenVINO NPU support not yet implemented")
}

// Minimal interface implementations to satisfy LlamaServer
// These will never be called since NewOpenVINOServer returns an error

func (s *openvinoServer) Ping(ctx context.Context) error {
	return fmt.Errorf("not implemented")
}

func (s *openvinoServer) WaitUntilRunning(ctx context.Context) error {
	return fmt.Errorf("not implemented")
}

func (s *openvinoServer) Completion(ctx context.Context, req CompletionRequest, fn func(CompletionResponse)) error {
	return fmt.Errorf("not implemented")
}

func (s *openvinoServer) Embedding(ctx context.Context, input string) ([]float32, error) {
	return nil, fmt.Errorf("not implemented")
}

func (s *openvinoServer) Tokenize(ctx context.Context, content string) ([]int, error) {
	return nil, fmt.Errorf("not implemented")
}

func (s *openvinoServer) Detokenize(ctx context.Context, tokens []int) (string, error) {
	return "", fmt.Errorf("not implemented")
}

func (s *openvinoServer) Close() error {
	return nil // No-op for now
}

func (s *openvinoServer) EstimatedVRAM() uint64 {
	return 0
}

func (s *openvinoServer) EstimatedTotal() uint64 {
	return 0
}

func (s *openvinoServer) EstimatedVRAMByGPU(gpuID string) uint64 {
	return 0
}

func (s *openvinoServer) Pid() int {
	return -1
}
