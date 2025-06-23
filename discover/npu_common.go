package discover

import (
	"log/slog"
	"os/exec"
	"strings"
)

// GetNPUInfo returns a list of detected NPU devices
func GetNPUInfo() []NpuInfo {
	var npus []NpuInfo

	// Try to detect NPUs using various methods
	npus = append(npus, detectIntelNPUs()...)
	npus = append(npus, detectAMDNPUs()...)

	if len(npus) > 0 {
		slog.Info("detected NPU devices", "count", len(npus))
	} else {
		slog.Debug("no NPU devices detected")
	}

	return npus
}

// detectIntelNPUs detects Intel NPU devices
func detectIntelNPUs() []NpuInfo {
	var npus []NpuInfo

	// Check for Intel NPU via lspci
	if hasIntelNPUDevice() {
		npu := NpuInfo{
			GpuInfo: GpuInfo{
				memInfo: memInfo{
					TotalMemory: 0, // NPU typically uses system RAM
					FreeMemory:  0, // Will be calculated dynamically
				},
				Library: "openvino",
				ID:      "intel_npu_0",
				Name:    "Intel NPU",
			},
			DeviceType:        "NPU",
			OpenVINOVersion:   getOpenVINOVersion(),
			QuantizationLevel: "INT4,INT8,FP16",
		}

		// Try to get more specific NPU information
		if npuName := getIntelNPUName(); npuName != "" {
			npu.Name = npuName
		}

		npus = append(npus, npu)
		slog.Debug("detected Intel NPU", "name", npu.Name, "id", npu.ID)
	}

	return npus
}

// detectAMDNPUs detects AMD XDNA NPU devices  
func detectAMDNPUs() []NpuInfo {
	var npus []NpuInfo

	// Check for AMD NPU via lspci
	if hasAMDNPUDevice() {
		npu := NpuInfo{
			GpuInfo: GpuInfo{
				memInfo: memInfo{
					TotalMemory: 0, // NPU typically uses system RAM
					FreeMemory:  0, // Will be calculated dynamically
				},
				Library: "openvino",
				ID:      "amd_npu_0",
				Name:    "AMD XDNA NPU",
			},
			DeviceType:        "NPU", 
			OpenVINOVersion:   getOpenVINOVersion(),
			QuantizationLevel: "INT4,INT8,FP16",
		}

		npus = append(npus, npu)
		slog.Debug("detected AMD NPU", "name", npu.Name, "id", npu.ID)
	}

	return npus
}

// hasIntelNPUDevice checks if Intel NPU is available via lspci
func hasIntelNPUDevice() bool {
	// Check for Intel Neural Processing Unit
	cmd := exec.Command("lspci")
	output, err := cmd.Output()
	if err != nil {
		slog.Debug("failed to run lspci", "error", err)
		return false
	}

	outputStr := strings.ToLower(string(output))
	
	// Look for Intel NPU indicators
	npuKeywords := []string{
		"neural",
		"npu", 
		"ai accelerator",
		"intel corp.*ai",
	}

	for _, keyword := range npuKeywords {
		if strings.Contains(outputStr, keyword) && strings.Contains(outputStr, "intel") {
			slog.Debug("found Intel NPU via lspci", "keyword", keyword)
			return true
		}
	}

	// Also check /sys/class/drm for Intel render devices (may include NPU)
	return hasIntelRenderDevice()
}

// hasAMDNPUDevice checks if AMD XDNA NPU is available
func hasAMDNPUDevice() bool {
	cmd := exec.Command("lspci")
	output, err := cmd.Output()
	if err != nil {
		slog.Debug("failed to run lspci", "error", err)
		return false
	}

	outputStr := strings.ToLower(string(output))
	
	// Look for AMD NPU indicators
	npuKeywords := []string{
		"xdna",
		"ai engine",
		"amd.*ai",
		"ryzen ai",
	}

	for _, keyword := range npuKeywords {
		if strings.Contains(outputStr, keyword) {
			slog.Debug("found AMD NPU via lspci", "keyword", keyword)
			return true
		}
	}

	return false
}

// getIntelNPUName attempts to get a more specific Intel NPU name
func getIntelNPUName() string {
	cmd := exec.Command("lspci", "-v")
	output, err := cmd.Output()
	if err != nil {
		return ""
	}

	lines := strings.Split(string(output), "\n")
	for _, line := range lines {
		lower := strings.ToLower(line)
		if strings.Contains(lower, "intel") && 
		   (strings.Contains(lower, "neural") || strings.Contains(lower, "npu") || strings.Contains(lower, "ai")) {
			// Try to extract a clean device name
			if parts := strings.SplitN(line, ":", 2); len(parts) > 1 {
				name := strings.TrimSpace(parts[1])
				if len(name) > 0 {
					return name
				}
			}
		}
	}
	
	return ""
}

// getOpenVINOVersion attempts to detect the available OpenVINO version
func getOpenVINOVersion() string {
	// Try Python OpenVINO first
	cmd := exec.Command("python3", "-c", "import openvino; print(openvino.__version__)")
	output, err := cmd.Output()
	if err == nil {
		version := strings.TrimSpace(string(output))
		if version != "" {
			slog.Debug("found OpenVINO Python version", "version", version)
			return version
		}
	}

	// Try python (fallback)
	cmd = exec.Command("python", "-c", "import openvino; print(openvino.__version__)")
	output, err = cmd.Output()
	if err == nil {
		version := strings.TrimSpace(string(output))
		if version != "" {
			slog.Debug("found OpenVINO Python version", "version", version)
			return version
		}
	}

	slog.Debug("OpenVINO not detected")
	return ""
}
