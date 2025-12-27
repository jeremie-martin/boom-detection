# Data Format Specification

> **Note**: Most users don't need this. Use `loader.py` which handles all decoding automatically.

Complete technical specification for simulation data files.

## simulation_data.bin

### Overview

Binary file containing compressed time-series data for all pendulums across all frames.

- **Format**: Custom binary with ZSTD compression
- **Endianness**: Little-endian throughout
- **Typical size**: 50-100 MB compressed (200-400 MB uncompressed)

### Header Structure

The header is exactly 144 bytes with no padding (packed structure):

```
Offset  Size    Type        Field                   Description
------  ----    ----        -----                   -----------
0       8       char[8]     magic                   "PNDL\x01\x00\x00\x00"
8       4       uint32      format_version          Always 2
12      4       uint32      pendulum_count          N pendulums (10,000-15,000)
16      4       uint32      frame_count             M frames (1,000-1,200)
20      8       float64     duration_seconds        Simulation duration
28      8       float64     max_dt                  Physics timestep limit
36      8       float64     gravity                 Gravitational constant (9.81)
44      8       float64     length1                 First arm length (meters)
52      8       float64     length2                 Second arm length (meters)
60      8       float64     mass1                   First mass (kg)
68      8       float64     mass2                   Second mass (kg)
76      8       float64     initial_angle1          Base angle for arm 1 (radians)
84      8       float64     initial_angle2          Base angle for arm 2 (radians)
92      8       float64     initial_velocity1       Base angular velocity 1 (rad/s)
100     8       float64     initial_velocity2       Base angular velocity 2 (rad/s)
108     8       float64     angle_variation         Spread applied to angles (radians)
116     4       uint32      floats_per_pendulum     Always 8
120     8       uint64      uncompressed_size       Payload size before compression
128     8       uint64      compressed_size         Payload size after compression
136     8       uint8[8]    reserved                Future use (zeros)
------
144 bytes total
```

### Payload Structure

The payload immediately follows the header and is ZSTD-compressed.

**Compression**: ZSTD level 3 (default)

**Decompressed layout**:

```
Total size: frame_count × pendulum_count × 8 × sizeof(float32)
          = frame_count × pendulum_count × 32 bytes

Memory layout (row-major, frame-first):
  data[frame][pendulum][field]

Where field index:
  0 = x1  (float32)  First joint X position
  1 = y1  (float32)  First joint Y position
  2 = x2  (float32)  Tip X position
  3 = y2  (float32)  Tip Y position
  4 = th1 (float32)  First arm angle (radians)
  5 = th2 (float32)  Second arm angle (radians)
  6 = w1  (float32)  First arm angular velocity (rad/s)
  7 = w2  (float32)  Second arm angular velocity (rad/s)
```

### Coordinate System

```
        (0, 0) pivot point
           │
           │ length1
           │
        (x1, y1) joint
           │
           │ length2
           │
        (x2, y2) tip
```

- Origin at pivot point (0, 0)
- Y-axis points downward (positive Y = below pivot)
- Angles measured from vertical (0 = hanging straight down)
- Positive angle = counterclockwise rotation

### Physical Constraints

Given the physics parameters, the data satisfies:

```
x1 = length1 × sin(th1)
y1 = length1 × cos(th1)
x2 = x1 + length2 × sin(th2)
y2 = y1 + length2 × cos(th2)
```

Note: Positions are stored redundantly (could be computed from angles) for convenience.

## metadata.json

### Overview

JSON file containing simulation configuration.

### Schema

```json
{
  "version": "string",           // Format version ("1.1")
  "created_at": "string",        // ISO 8601 timestamp

  "physics": {
    "gravity": "number",         // m/s² (9.81)
    "length1": "number",         // meters (1.0)
    "length2": "number",         // meters (1.0)
    "mass1": "number",           // kg (1.0)
    "mass2": "number",           // kg (1.0)
    "initial_angle1_deg": "number",  // degrees from vertical
    "initial_angle2_deg": "number",  // degrees from vertical
    "initial_velocity1": "number",   // rad/s (0.0)
    "initial_velocity2": "number"    // rad/s (0.0)
  },

  "simulation": {
    "pendulum_count": "integer", // 10,000-15,000
    "angle_variation_deg": "number",  // typically 0.1°
    "duration_seconds": "number",     // typically 18s
    "total_frames": "integer",        // 1,000-1,200
    "physics_quality": "string",      // "high"
    "max_dt": "number",               // physics timestep limit
    "substeps": "integer",            // integration substeps
    "dt": "number"                    // actual timestep
  }
}
```

**Note**: The `results.boom_frame` field in metadata (if present) is from unreliable automated detection. Use `annotations.json` for ground truth labels.

## annotations.json

### Schema

```json
{
  "version": 2,
  "target_defs": {
    "boom": "frame",           // Label type: frame number
    "boom_quality": "score"    // Label type: 0-1 score
  },
  "annotations": [
    {
      "id": "string",          // Unique run identifier
      "data_path": "string",   // Relative path to simulation_data.bin
      "targets": {
        "boom": "number",      // Frame number (integer stored as float)
        "boom_quality": "number"  // Quality score 0.0-1.0
      },
      "notes": "string"        // Optional annotator notes
    }
  ]
}
```

### Label Definitions

**boom** (frame):
- Type: Integer (stored as float in JSON)
- Range: Typically 200-950
- Definition: The frame where chaotic divergence becomes visually significant
- Annotation method: Human visual inspection of rendered video

**boom_quality** (score):
- Type: Float
- Range: 0.0 to 1.0
- Definition: Subjective rating of visual impressiveness
- 0.0 = Poor (weak, gradual, or late boom)
- 0.5 = Average (clear but unremarkable)
- 1.0 = Excellent (dramatic, well-timed, visually striking)

## File Size Reference

Typical sizes for one simulation:

| File | Typical Size | Notes |
|------|--------------|-------|
| simulation_data.bin | 50-100 MB | Compressed |
| simulation_data.bin | 200-400 MB | Uncompressed in memory |
| metadata.json | 2-3 KB | |

Total dataset: ~12 GB for 50 simulations (binary data only)
