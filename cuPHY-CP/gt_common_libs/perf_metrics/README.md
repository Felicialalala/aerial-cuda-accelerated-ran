# Performance Metrics Library

A high-performance timing and metrics collection library for cuPHY modules, designed for zero-overhead performance measurement with NVLOG integration.

## Features

- **Zero Dynamic Allocation**: No memory allocation during timing operations (only during initialization)
- **High-Precision Timing**: Uses `std::chrono::system_clock` for nanosecond-precision measurements
- **NVLOG Integration**: Direct integration with NVLOG system for structured logging
- **Compile-Time Safety**: Template-based interface with compile-time NVLOG tag validation
- **Single-Threaded Design**: Optimized for single-threaded usage (not thread-safe)
- **Minimal Interface**: Only 4 public methods for simplicity

## Quick Start

```cpp
#include "perf_metrics/perf_metrics_accumulator.hpp"

// Define your NVLOG tag (must be compile-time constant)
#define TAG_MY_PERF 54  // Example: NVIPC.TIMING

// Create accumulator with pre-registered sections
perf_metrics::PerfMetricsAccumulator pma{"Section 1", "Section 2"};

// Time your code sections
for (int i = 0; i < 100; i++) {
    pma.startSection("Section 1");
    // ... your code here ...
    pma.stopSection("Section 1");
    
    pma.startSection("Section 2");
    // ... your code here ...
    pma.stopSection("Section 2");
}

// Log accumulated results
pma.logDurations<TAG_MY_PERF>();  // Uses INFO level by default
pma.reset();  // Clear for next measurement cycle
```

## API Reference

### Constructor

```cpp
// Pre-register sections using initializer list
PerfMetricsAccumulator(const std::initializer_list<const char*>& sectionNames);

// Default constructor (sections must be pre-registered)
PerfMetricsAccumulator();
```

### Core Methods

```cpp
// Start timing a section
void startSection(const char* sectionName);

// Stop timing a section
void stopSection(const char* sectionName);

// Log accumulated durations with specified NVLOG tag and level
template<int NvlogTag, LogLevel Level = LogLevel::INFO>
void logDurations() const;

// Reset all accumulated data to zero
void reset();
```

### Log Levels

```cpp
enum class LogLevel {
    VERBOSE,  // NVLOGV_FMT
    DEBUG,    // NVLOGD_FMT  
    INFO,     // NVLOGI_FMT (default)
    WARN,     // NVLOGW_FMT
    ERROR     // NVLOGE_FMT (uses AERIAL_CUPHY_EVENT)
};
```

## Usage Examples

### Basic Usage

```cpp
#include "perf_metrics/perf_metrics_accumulator.hpp"

// Define NVLOG tags for your module
#define NVLOG_TAG_BASE_MY_MODULE 1300
#define TAG_MY_PERF_METRICS (NVLOG_TAG_BASE_MY_MODULE + 1)

void example_usage() {
    // IMPORTANT: Pre-register ALL sections to avoid dynamic allocation
    perf_metrics::PerfMetricsAccumulator pma{"Section 1", "Section 2"};
    
    // Simulate work with timing
    for (int ii = 0; ii < 100; ii++) {
        pma.startSection("Section 1");  // Must be pre-registered
        // ... code for Section 1 ...
        pma.stopSection("Section 1");
        
        pma.startSection("Section 2");  // Must be pre-registered  
        // ... code for Section 2 ...
        pma.stopSection("Section 2");
    }
    
    // Log results with different levels
    pma.logDurations<TAG_MY_PERF_METRICS>();                                  // INFO level (default)
    pma.logDurations<TAG_MY_PERF_METRICS, perf_metrics::LogLevel::DEBUG>();  // DEBUG level
    pma.logDurations<TAG_MY_PERF_METRICS, perf_metrics::LogLevel::VERBOSE>(); // VERBOSE level
    
    // Reset for next measurement cycle
    pma.reset();
}
```

### RU Emulator Example

```cpp
#define TAG_RU_PERF (NVLOG_TAG_BASE_RU_EMULATOR + 15)

void ru_emulator_example() {
    perf_metrics::PerfMetricsAccumulator pma{"UL Processing", "DL Processing", "BFW Generation"};
    
    // Time UL processing
    pma.startSection("UL Processing");
    // ... UL processing code ...
    pma.stopSection("UL Processing");
    
    // Time DL processing  
    pma.startSection("DL Processing");
    // ... DL processing code ...
    pma.stopSection("DL Processing");
    
    // Time BFW generation
    pma.startSection("BFW Generation");
    // ... BFW generation code ...
    pma.stopSection("BFW Generation");
    
    // Log with ru-emulator specific tag
    pma.logDurations<TAG_RU_PERF, perf_metrics::LogLevel::INFO>();
}
```

### Clean Usage with Namespace

```cpp
void clean_usage_example() {
    using namespace perf_metrics;
    
    PerfMetricsAccumulator pma{"Section 1", "Section 2"};
    
    // Only 4 methods available: startSection, stopSection, logDurations, reset
    pma.startSection("Section 1");
    // ... timing code ...
    pma.stopSection("Section 1");
    
    pma.logDurations<TAG_MY_PERF_METRICS>();                    // INFO level (default)
    pma.logDurations<TAG_MY_PERF_METRICS, LogLevel::DEBUG>();  // DEBUG level
    
    pma.reset();  // Clear all accumulated data
}
```

## Output Format

The library outputs timing data in a compact, single-line format:

```
PerfMetricsAccumulator - Section1:100:1542,Section2:50:1700
```

Where:
- `Section1:100:1542` means "Section1" was called 100 times and took 1542 microseconds total
- `Section2:50:1700` means "Section2" was called 50 times and took 1700 microseconds total  
- Multiple sections are comma-separated
- Format: `SectionName:count:duration_us`
- Times are reported in microseconds (μs)
- Internal precision is nanoseconds for accuracy

## Error Handling

The library uses `printf` for error reporting:

```cpp
// Starting non-existent section
pma.startSection("Unknown");
// Output: PerfMetricsAccumulator: Error - section 'Unknown' not found. Pre-register all sections in constructor.

// Double start
pma.startSection("Section1");
pma.startSection("Section1");  
// Output: PerfMetricsAccumulator: Error - timing already active for section 'Section1'

// Stop without start
pma.stopSection("Section1");
// Output: PerfMetricsAccumulator: Error - timing not active for section 'Section1'
```

## Performance Characteristics

- **Zero allocation during timing**: All memory allocated at construction time
- **Fixed-size logging buffer**: 1024 bytes, no dynamic allocation for log formatting
- **Nanosecond precision**: Uses `std::chrono::system_clock` for high-precision timing
- **Microsecond output**: Converted for readability while maintaining precision
- **Compile-time optimization**: Template-based logging resolves at compile time

## Integration

### CMake

The library is automatically available when linking with `perf_metrics`:

```cmake
target_link_libraries(your_target PRIVATE perf_metrics)
```

### Dependencies

- **nvlog**: For structured logging output
- **C++20**: Uses modern C++ features like `if constexpr`

## Thread Safety

Currently designed for single-threaded usage. Each thread should have its own `PerfMetricsAccumulator` instance if multi-threading is needed.

## Best Practices

1. **Pre-register all sections** in the constructor to avoid runtime allocation
2. **Use meaningful section names** that clearly identify the code being timed
3. **Choose appropriate NVLOG tags** that follow your module's tag conventions
4. **Reset regularly** to avoid overflow of accumulated values
5. **Use consistent log levels** for similar types of measurements
6. **Keep section names short** to fit within the 1024-byte log buffer

## Testing

Run the unit test to verify functionality:

```bash
ninja perf_metrics_accumulator_unit_test
./cuPHY-CP/gt_common_libs/perf_metrics/test/perf_metrics_accumulator_unit_test
```

The test covers:
- Basic timing functionality
- Error handling scenarios  
- Different log levels
- Reset functionality
- NVLOG integration
