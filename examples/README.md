# Orientation Fusion Examples

This directory contains examples demonstrating the `OrientationFusion` class, which provides sensor fusion capabilities for combining gyroscope and ArUco marker data for robust orientation tracking.

## Overview

The `OrientationFusion` class implements a two-stage filtering approach:

1. **Complementary Filter**: Combines high-frequency gyroscope data with low-frequency ArUco marker data
2. **Low-Pass Filter**: Reduces noise in the final output

## Files

### Core Class
- `../src/gino/kinematics/orientation_fusion.py` - The main `OrientationFusion` class

### Examples
- `orientation_fusion_example.py` - Full real-time example with camera and MCU hardware
- `simple_orientation_fusion_example.py` - Simple example with simulated data (no hardware required)

## Quick Start

### Simple Example (No Hardware Required)

```bash
cd examples
python simple_orientation_fusion_example.py
```

This example:
- Uses simulated sensor data
- Demonstrates the fusion algorithm
- Shows plots comparing true, gyroscope, ArUco, and filtered data
- Calculates RMSE performance metrics

### Full Example (Requires Hardware)

```bash
cd examples
python orientation_fusion_example.py
```

This example requires:
- USB camera
- Arduino MCU with gyroscope sensor
- ArUco marker cube

## Using the OrientationFusion Class

### Basic Usage

```python
from src.gino.kinematics.orientation_fusion import OrientationFusion

# Initialize the fusion system
fusion = OrientationFusion(
    complementary_alpha=0.90,  # 90% gyro, 10% ArUco
    lpf_alpha=0.35,           # Low-pass filter strength
    dt=0.010                  # Time step in seconds
)

# Update with sensor data
fusion.update_gyro_data(roll_velocity, pitch_velocity, yaw_velocity)
fusion.update_aruco_data(roll_angle, pitch_angle, yaw_angle)  # Can be None

# Get filtered orientation
filtered_roll, filtered_pitch, filtered_yaw = fusion.process()
```

### Parameters

#### Complementary Filter (`complementary_alpha`)
- **Range**: 0.0 to 1.0
- **Higher values** (0.9-0.98): More weight to gyroscope, better for fast movements
- **Lower values** (0.7-0.9): More weight to ArUco, better for absolute accuracy
- **Recommended**: 0.90-0.95 for most applications

#### Low-Pass Filter (`lpf_alpha`)
- **Range**: 0.0 to 1.0
- **Higher values** (0.5-1.0): Less smoothing, more responsive
- **Lower values** (0.1-0.5): More smoothing, less noise
- **Recommended**: 0.3-0.5 for noise reduction

#### Time Step (`dt`)
- **Units**: Seconds
- **Should match** your actual sensor update rate
- **Typical values**: 0.01 (100Hz) to 0.033 (30Hz)

### Advanced Usage

```python
# Get current state for debugging/monitoring
state = fusion.get_current_state()
print(f"Gyro: {state['gyro']}")
print(f"ArUco: {state['aruco']}")
print(f"Filtered: {state['filtered']}")

# Convert rotation matrix to Euler angles
rotation_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
roll, pitch, yaw = OrientationFusion.rotation_matrix_to_euler(rotation_matrix)
```

## How It Works

### Sensor Fusion Algorithm

1. **Gyroscope Integration**: Angular velocities are integrated over time to estimate orientation
2. **Complementary Filter**: Combines gyroscope (high frequency) with ArUco (low frequency)
3. **Low-Pass Filter**: Smooths the output to reduce noise

### Mathematical Details

#### Complementary Filter
```
filtered_angle = α * (previous_angle + gyro_change) + (1-α) * aruco_angle
```

#### Low-Pass Filter
```
filtered_output = α * current_input + (1-α) * previous_output
```

## Performance Tuning

### For Fast Movements
- Increase `complementary_alpha` to 0.95-0.98
- Decrease `lpf_alpha` to 0.2-0.3

### For High Accuracy
- Decrease `complementary_alpha` to 0.8-0.9
- Increase `lpf_alpha` to 0.4-0.6

### For Noisy Environments
- Decrease `lpf_alpha` to 0.1-0.3
- Keep `complementary_alpha` moderate (0.85-0.92)

## Troubleshooting

### Common Issues

1. **Drift in filtered output**
   - Decrease `complementary_alpha` to rely more on ArUco
   - Check ArUco detection reliability

2. **Laggy response**
   - Increase `lpf_alpha` for less smoothing
   - Increase `complementary_alpha` for more gyroscope weight

3. **Noisy output**
   - Decrease `lpf_alpha` for more smoothing
   - Check sensor calibration

### Debugging

Use the `get_current_state()` method to monitor individual sensor contributions:

```python
state = fusion.get_current_state()
if state['aruco']['roll'] is None:
    print("Warning: No ArUco detection")
if abs(state['gyro']['roll']) > 100:
    print("Warning: High gyroscope values")
```

## Dependencies

- `numpy` - Numerical computations
- `matplotlib` - Plotting (for examples)
- `opencv-python` - Camera interface (for full example)
- `src.gino.aruco.cube_detection` - ArUco detection (for full example)
- `src.gino.mcu.mcu` - MCU communication (for full example) 