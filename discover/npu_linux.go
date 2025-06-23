package discover

import (
	"os"
	"path/filepath"
	"strings"
)

// hasIntelRenderDevice checks for Intel render devices in /sys/class/drm
func hasIntelRenderDevice() bool {
	drmPath := "/sys/class/drm"
	
	entries, err := os.ReadDir(drmPath)
	if err != nil {
		return false
	}

	for _, entry := range entries {
		if strings.HasPrefix(entry.Name(), "renderD") {
			// Check if this render device is Intel
			devicePath := filepath.Join(drmPath, entry.Name(), "device", "vendor")
			vendorData, err := os.ReadFile(devicePath)
			if err != nil {
				continue
			}
			
			vendor := strings.TrimSpace(string(vendorData))
			// Intel PCI vendor ID is 0x8086
			if vendor == "0x8086" {
				return true
			}
		}
	}
	
	return false
}
