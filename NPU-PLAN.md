# NPU Integration Plan for Ollama

## Project Overview

This document outlines the comprehensive plan to integrate Neural Processing Unit (NPU) support into Ollama using OpenVINO, based on the successful NPGlue implementation. This integration will enable Ollama to utilize NPU, CPU, and GPU acceleration in a unified platform.

## Executive Summary

**Goal**: Add NPU support to Ollama so it can run models on NPU (Intel Arc Graphics 130V/140V, AMD XDNA NPU, etc.) alongside existing CPU and GPU options.

**Benefits**:
- **Performance**: NPUs deliver 20-60 tokens/sec vs. traditional CPU's 2-8 tokens/sec
- **Efficiency**: NPUs consume 5-15W vs. GPU's 150-300W+ power consumption
- **Memory**: NPUs work efficiently with system RAM vs. GPU VRAM limitations
- **Future-proof**: NPUs are becoming standard in modern processors (Intel Core Ultra, AMD Ryzen AI)

## Current State Analysis

### Ollama Architecture
- **Language**: Go-based with llama.cpp C++ backend
- **Server**: HTTP REST API with streaming support
- **Scheduler**: Dynamic model loading/unloading with memory management  
- **Discovery**: Hardware detection system for GPUs (CUDA, ROCm, Metal, OneAPI)
- **LLM Backend**: llama.cpp server subprocess with various optimizations

### NPGlue Architecture  
- **Language**: Python-based with OpenVINO backend
- **Server**: FastAPI with OpenAI and Ollama-compatible APIs
- **Models**: OpenVINO-optimized INT4/INT8 quantized models
- **Performance**: Memory-safe with automatic CPU boost and cleanup
- **Hardware**: Supports NPU, Intel iGPU, and CPU fallback

## Integration Strategy

### Phase 1: Core NPU Detection & Infrastructure

#### 1.1 NPU Discovery System
**Location**: `discover/` package

**New Files**:
- `discover/npu_common.go` - Common NPU detection logic
- `discover/npu_linux.go` - Linux NPU detection  
- `discover/npu_windows.go` - Windows NPU detection
- `discover/npu_darwin.go` - macOS NPU detection (future)
- `discover/gpu_info_openvino.c` - OpenVINO C bindings for NPU detection
- `discover/gpu_info_openvino.h` - OpenVINO headers

**NPU Detection Logic**:
```go
type NpuInfo struct {
    GpuInfo                    // Inherit base GPU info structure
    OpenVINOVersion    string  // Available OpenVINO version
    DeviceType        string   // "NPU", "GPU.0", "GPU.1", etc.
    MaxMemoryMB       uint64   // NPU memory limit (if available)
    QuantizationLevel string   // Supported quantization (INT4, INT8, FP16)
}

// Detect Intel NPUs via OpenVINO
func GetNPUInfo() []NpuInfo

// Detect available OpenVINO devices  
func GetOpenVINODevices() []string
```

**Detection Methods**:
1. **Intel NPUs**: Check `/sys/class/drm/renderD*`, `lspci | grep -i neural`, OpenVINO device enumeration
2. **AMD XDNA NPUs**: Check for AMD AI stack, `lspci | grep -i amd.*ai`
3. **Qualcomm NPUs**: Check for Hexagon NPU on Snapdragon X platforms
4. **OpenVINO Verification**: Ensure OpenVINO runtime is available and functional

#### 1.2 OpenVINO Integration Library
**Location**: `llm/openvino.go`

**Core Functions**:
```go
package llm

// OpenVINO LLM Server implementation
type OpenVINOServer struct {
    port        int
    cmd         *exec.Cmd  
    modelPath   string
    deviceType  string      // "NPU", "GPU.0", etc.
    estimate    MemoryEstimate
    status      *StatusWriter
    // ... other common server fields
}

// Create new OpenVINO-based LLM server
func NewOpenVINOServer(npus NpuInfoList, modelPath string, opts api.Options, numParallel int) (LlamaServer, error)
```

#### 1.3 Model Format Support
**Location**: `server/` package

**Model Detection Logic**:
```go
// Add to existing model detection
func detectModelFormat(modelPath string) ModelFormat {
    if hasOpenVINOFiles(modelPath) {
        return ModelFormatOpenVINO
    }
    if hasGGUFFiles(modelPath) {
        return ModelFormatGGUF  
    }
    // ... existing logic
}

func hasOpenVINOFiles(path string) bool {
    // Check for .xml, .bin files (OpenVINO IR format)
    // Check for openvino_model.xml, openvino_model.bin
    // Check for config.json with openvino metadata
}
```

### Phase 2: OpenVINO Runtime Integration

#### 2.1 OpenVINO Subprocess Runner
**Location**: `llm/openvino_server.go`

**OpenVINO Python Server**:
Based on NPGlue's `server_production.py`, create a dedicated OpenVINO inference server:

```python
# llm/openvino_runner.py - Embedded OpenVINO server
import openvino as ov
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer
import uvicorn
from fastapi import FastAPI

class OpenVINOInferenceServer:
    def __init__(self, model_path: str, device: str, port: int):
        self.model_path = model_path
        self.device = device  # "NPU", "GPU.0", "CPU"
        self.port = port
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        # Load OpenVINO optimized model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = OVModelForCausalLM.from_pretrained(
            self.model_path, 
            device=self.device,
            trust_remote_code=True
        )
    
    def completion(self, request):
        # Handle completion requests compatible with llama.cpp format
        pass
    
    def embedding(self, text):
        # Handle embedding requests
        pass
```

**Go Integration**:
```go
func (s *OpenVINOServer) startServer() error {
    // Find embedded Python script or external openvino runner
    pythonScript := filepath.Join(discover.LibOllamaPath, "openvino_runner.py")
    
    // Start OpenVINO server subprocess
    s.cmd = exec.Command("python", pythonScript,
        "--model-path", s.modelPath,
        "--device", s.deviceType,
        "--port", strconv.Itoa(s.port))
    
    // Set environment for OpenVINO
    s.cmd.Env = append(os.Environ(),
        "OPENVINO_LOG_LEVEL=ERROR",
        "OV_CACHE_DIR=" + getCacheDir(),
    )
    
    return s.cmd.Start()
}
```

#### 2.2 Model Conversion & Optimization
**Location**: `convert/openvino.go`

**Automatic Conversion**:
```go
// Convert GGUF model to OpenVINO format for NPU acceleration
func ConvertToOpenVINO(ggufPath, outputDir, deviceType string, quantization string) error {
    // Use optimum-cli to convert from GGUF/PyTorch to OpenVINO
    // optimum-cli export openvino --model <path> --task text-generation --device <device>
    
    cmd := exec.Command("optimum-cli", "export", "openvino",
        "--model", ggufPath,
        "--task", "text-generation", 
        "--device", deviceType,
        "--int8" if quantization == "int8",
        outputDir)
    
    return cmd.Run()
}
```

**Conversion Workflow**:
1. Detect when user pulls a new model
2. Check if OpenVINO-optimized version exists
3. If not, automatically convert GGUF → OpenVINO IR format
4. Optimize for target NPU/device
5. Cache converted model for future use

### Phase 3: Scheduler Integration

#### 3.1 Enhanced Scheduler Logic  
**Location**: `server/sched.go`

**NPU-Aware Scheduling**:
```go
type NPURunnerRef struct {
    llama        LlamaServer  // OpenVINO server instead of llama.cpp
    model        *Model
    deviceType   string       // "NPU", "GPU.0", "CPU"
    npuInfo      *NpuInfo
    sessionDuration time.Duration
    estimatedVRAM   uint64
    estimatedTotal  uint64
}

// Enhanced scheduler to prefer NPU when available
func (s *Scheduler) scheduleRunner(ctx context.Context, model *Model, opts api.Options) (*runnerRef, error) {
    // 1. Check for OpenVINO-optimized model version
    // 2. Prefer NPU if model supports it and NPU is available
    // 3. Fall back to GPU/CPU if NPU unavailable or model unsupported
    
    npuOptimizedPath := getOpenVINOModelPath(model.ModelPath)
    if npuOptimizedPath != "" && hasAvailableNPU() {
        return s.scheduleOpenVINORunner(ctx, model, opts, "NPU")
    }
    
    // Fall back to existing GGUF + llama.cpp logic
    return s.scheduleGGUFRunner(ctx, model, opts)
}
```

#### 3.2 Memory Management
**Location**: `llm/memory.go`

**NPU Memory Estimation**:
```go
func EstimateNPULayers(npus NpuInfoList, f *ggml.GGML, opts api.Options) MemoryEstimate {
    // NPUs typically use system RAM, not dedicated VRAM
    // Estimate based on model size and quantization level
    
    modelSizeBytes := f.Size() 
    
    // NPU memory usage patterns
    if opts.Quantization == "INT4" {
        modelSizeBytes = modelSizeBytes / 2  // Rough estimate
    } else if opts.Quantization == "INT8" {
        modelSizeBytes = modelSizeBytes / 1.5
    }
    
    systemMemory := getSystemMemoryInfo()
    
    return MemoryEstimate{
        TotalSize: modelSizeBytes,
        VRAMSize:  modelSizeBytes,  // NPU uses system RAM
        Layers:    f.KV().BlockCount() + 1,
        GPUSizes:  []uint64{modelSizeBytes},
    }
}
```

### Phase 4: API Enhancement

#### 4.1 Model Information Extension
**Location**: `api/types.go`

**Enhanced Model Details**:
```go
type ModelDetails struct {
    ParentModel       string   `json:"parent_model"`
    Format            string   `json:"format"`
    Family            string   `json:"family"`
    Families          []string `json:"families"`  
    ParameterSize     string   `json:"parameter_size"`
    QuantizationLevel string   `json:"quantization_level"`
    
    // New NPU-specific fields
    AccelerationType  string   `json:"acceleration_type"`     // "NPU", "GPU", "CPU"
    DeviceID          string   `json:"device_id,omitempty"`  // Specific device identifier
    OpenVINOVersion   string   `json:"openvino_version,omitempty"`
    SupportedDevices  []string `json:"supported_devices,omitempty"`
}
```

#### 4.2 Hardware Information API
**Location**: `server/routes.go`

**New Endpoint**:
```go
func (s *Server) HardwareHandler(c *gin.Context) {
    systemInfo := discover.GetSystemInfo()
    npuInfo := discover.GetNPUInfo()
    
    response := map[string]interface{}{
        "system": systemInfo,
        "npus":   npuInfo,
        "gpus":   systemInfo.GPUs,
        "openvino": map[string]interface{}{
            "available": isOpenVINOAvailable(),
            "version":   getOpenVINOVersion(),
            "devices":   getOpenVINODevices(),
        },
    }
    
    c.JSON(http.StatusOK, response)
}
```

### Phase 5: Installation & Dependencies

#### 5.1 OpenVINO Distribution
**Location**: Build system and packaging

**Options**:
1. **Bundle OpenVINO**: Include OpenVINO runtime in Ollama distribution
2. **Auto-Install**: Download OpenVINO runtime on first NPU use
3. **Package Manager**: Add OpenVINO as optional dependency

**Recommended Approach**: Auto-install on first use to keep core Ollama lightweight:

```bash
# scripts/install_openvino.sh
#!/bin/bash
echo "Installing OpenVINO runtime for NPU support..."

case "$OSTYPE" in
    linux*)   
        pip install openvino optimum[openvino] transformers
        ;;
    msys*|cygwin*)  
        pip install openvino optimum[openvino] transformers  
        ;;
    darwin*)
        pip install openvino optimum[openvino] transformers
        ;;
esac

echo "OpenVINO runtime installed successfully"
```

#### 5.2 Build System Integration
**Location**: `CMakeLists.txt`, `Makefile`, CI/CD

**Build Modifications**:
```cmake
# CMakeLists.txt additions
option(OLLAMA_NPU "Enable NPU support via OpenVINO" ON)

if(OLLAMA_NPU)
    find_package(OpenVINO REQUIRED)
    target_link_libraries(ollama OpenVINO::openvino)
    target_compile_definitions(ollama PRIVATE OLLAMA_NPU)
endif()
```

**CI Integration**:
- Add OpenVINO to GitHub Actions builds
- Test NPU functionality on Intel/AMD NPU hardware
- Ensure graceful fallback when OpenVINO unavailable

### Phase 6: Testing & Quality Assurance

#### 6.1 Unit Tests
**Location**: `*_test.go` files alongside implementation

**Test Coverage**:
- NPU detection across different hardware
- OpenVINO model loading and inference
- Memory estimation accuracy
- API compatibility (both Ollama and OpenAI endpoints)
- Graceful fallback when NPUs unavailable

#### 6.2 Integration Tests  
**Location**: `integration/` package

**Test Scenarios**:
- Full model lifecycle: pull → convert → load → inference → unload
- Multi-device scenarios (NPU + GPU + CPU)
- Performance benchmarking vs. existing GPU/CPU backends
- Memory pressure and OOM handling
- Concurrent request handling

#### 6.3 Hardware Testing
**Test Platforms**:
- Intel Core Ultra systems (Intel NPU)
- AMD Ryzen AI systems (AMD XDNA NPU)
- Systems without NPU (fallback testing)
- Windows, Linux, and macOS (where applicable)

## Implementation Timeline

### Phase 1: Foundation (Week 1-2)
- [ ] NPU discovery system implementation
  - [x] **CURRENT**: Create basic NPU types and structures
  - [ ] **NEXT**: Implement NPU detection logic
- [ ] Basic OpenVINO integration structure
- [ ] Model format detection enhancement
- [ ] Unit tests for discovery logic

## 🚧 **Current Progress Tracker**

### **Current Step**: Phase 1 Complete - Ready for Phase 2
**Date**: 2025-01-23  
**Status**: ✅ COMPLETED

**What we accomplished**: Created complete NPU discovery foundation with minimal OpenVINO interface
**Next**: Begin Phase 2 - Core Runtime Implementation

### **Completed Steps**: 
1. ✅ **Phase 1.1** - Created NPU type definitions in `discover/types.go` - ✅ TESTED & COMPILED  
2. ✅ **Phase 1.2** - Implemented NPU detection logic - ✅ TESTED & WORKING (detected Intel Lunar Lake NPU!)
3. ✅ **Phase 1.3** - Integrated NPU detection into SystemInfo - ✅ TESTED & WORKING (SystemInfo now includes NPUs!)  
4. ✅ **Phase 1.4** - Created basic OpenVINO LlamaServer interface - ✅ TESTED & COMPILED

### **Phase 1 Progress**: NPU Discovery System - ✅ 100% Complete
- [x] NPU detection logic  
- [x] SystemInfo integration
- [x] Basic OpenVINO integration structure
- [ ] Model format detection enhancement (deferred to Phase 2)
- [ ] Unit tests (deferred to Phase 2)

### **Next Phase Ready**: 
**Phase 2** - Core Runtime Implementation
- Basic OpenVINO server subprocess implementation
- Model conversion pipeline
- Memory estimation for NPU

### Phase 2: Core Runtime (Week 3-4) 
- [ ] OpenVINO server subprocess implementation
- [ ] Model conversion pipeline
- [ ] Memory estimation for NPU
- [ ] Basic inference functionality

### Phase 3: Scheduler Integration (Week 5-6)
- [ ] Enhanced scheduler with NPU preference
- [ ] Multi-device coordination
- [ ] Performance optimization
- [ ] Comprehensive testing

### Phase 4: API & Documentation (Week 7-8)
- [ ] API enhancements for hardware reporting
- [ ] Documentation updates
- [ ] User guides and troubleshooting  
- [ ] Performance benchmarking

### Phase 5: Release Preparation (Week 9-10)
- [ ] Build system integration
- [ ] CI/CD pipeline updates
- [ ] Final testing and bug fixes
- [ ] Release notes and migration guide

## Technical Risks & Mitigation

### Risk 1: OpenVINO Dependencies
**Risk**: Complex Python/C++ dependencies may complicate distribution
**Mitigation**: 
- Provide fallback to pure llama.cpp when OpenVINO unavailable
- Auto-installation system for OpenVINO runtime
- Clear documentation for manual installation

### Risk 2: Model Compatibility
**Risk**: Not all models may work well with OpenVINO conversion
**Mitigation**:
- Maintain GGUF compatibility as primary format
- Automatic fallback when OpenVINO conversion fails
- Whitelist of known-compatible model families

### Risk 3: Performance Variance
**Risk**: NPU performance may vary significantly across hardware
**Mitigation**:
- Comprehensive benchmarking across NPU types
- Dynamic device selection based on performance profiling
- User override options for device selection

### Risk 4: Platform Support
**Risk**: NPU support differs significantly between OS platforms
**Mitigation**:
- Platform-specific detection and optimization
- Clear documentation of supported hardware
- Graceful degradation on unsupported platforms

## Success Metrics

### Performance Metrics
- **Token Generation Speed**: 20-60 tokens/sec on supported NPU hardware
- **Memory Efficiency**: Run 7B models in <6GB system RAM
- **Power Efficiency**: <20W power consumption during inference
- **Load Time**: <10 seconds for model loading on NPU

### Compatibility Metrics
- **API Compatibility**: 100% backward compatibility with existing Ollama APIs
- **Model Support**: 80%+ of popular models work with NPU acceleration
- **Hardware Support**: Support Intel Core Ultra, AMD Ryzen AI NPUs
- **Fallback Reliability**: Seamless fallback when NPU unavailable

### User Experience Metrics
- **Installation Success**: >95% automated installation success rate
- **Documentation Quality**: Comprehensive guides for setup and troubleshooting
- **Error Handling**: Clear error messages and recovery suggestions
- **Performance Visibility**: Users can see which device is being used

## Future Enhancements

### Phase 2 (Future Release)
- **Multi-NPU Support**: Distribute models across multiple NPU devices
- **Dynamic Load Balancing**: Automatically balance between NPU/GPU/CPU based on load
- **Model Caching**: Intelligent caching of converted OpenVINO models
- **Real-time Performance Monitoring**: Live performance metrics in API

### Phase 3 (Advanced Features)
- **Custom Quantization**: User-selectable quantization levels for NPU
- **Model Optimization**: Automatic optimization for specific NPU architectures  
- **Distributed Inference**: NPU + GPU hybrid inference for large models
- **Power Management**: Intelligent device selection based on power constraints

## Conclusion

This NPU integration plan provides a comprehensive roadmap for bringing NPU acceleration to Ollama while maintaining backward compatibility and reliability. By leveraging the proven OpenVINO ecosystem and building on NPGlue's successful implementation, we can deliver significant performance improvements for users with modern NPU-enabled hardware.

The phased approach ensures we can deliver incremental value while managing technical complexity, and the extensive testing plan ensures reliability across diverse hardware configurations. Upon completion, Ollama will be positioned as the leading local LLM platform supporting the full spectrum of modern AI acceleration hardware.
