# Gesture Control Map Navigation

A real-time interactive map navigation system using hand gesture recognition with MediaPipe and OpenGL. Control a 2D map through natural hand movements - pan the map with your left hand and zoom with pinch gestures from your right hand.

## Features

- 🎮 **Hand Gesture Recognition**: Real-time hand tracking using MediaPipe
- 🗺️ **Interactive Map**: Grid-based map with cities and landmarks
- 🖐️ **Gesture Controls**:
  - **Left Hand**: Pan the map (move hand to navigate)
  - **Right Hand Pinch**: Zoom in/out (spread fingers to zoom in, pinch to zoom out)
- 📍 **Map Features**: Multiple cities, roads, and coordinate axes visualization
- 🎯 **Smooth Navigation**: Responsive gesture detection with zoom bounds

## Requirements

- Python 3.8+
- OpenGL-capable graphics card
- Webcam

## Installation

```bash
pip install -r requirements.txt
```

### Dependencies

- `opencv-python`: Video capture and processing
- `mediapipe`: Hand detection and tracking
- `pygame`: OpenGL window management
- `PyOpenGL`: 3D graphics rendering
- `numpy`: Numerical computations

## Usage

Run the application:

```bash
python gesture_3d_control.py
```

### Controls

**Left Hand:**
- Move your hand in any direction → **Pan the map** (navigate around)

**Right Hand:**
- Thumb + Index finger apart → **Zoom in**
- Thumb + Index finger together (pinch) → **Zoom out**
- Zoom range: 0.5x to 3.0x

**Other:**
- Press 'q' or close window → Exit application

## How It Works

### Hand Tracking
- Uses MediaPipe's hand tracking solution to detect both hands in real-time
- Tracks 21 landmark points per hand for precise gesture recognition

### Map Navigation
- **Panning**: Left hand X/Y position controls map offset
- **Zooming**: Distance between thumb and index finger on right hand controls zoom level
- Grid-based map with visual landmarks (cities)

### Map Features
- Grid background for reference
- Nine cities positioned across the map
- Roads connecting cities
- X/Y coordinate axes at the center
- Dynamic camera that scales and pans based on gestures

## Map Structure

The map displays:
- **Central Grid**: Reference grid lines for orientation
- **Cities**: Orange spheres representing major locations
  - Central, North East, North West, South East, South West
  - East, West, North, South directions
- **Roads**: Yellow lines connecting cities
- **Axes**: Red (X-axis) and Green (Y-axis) for coordinate reference

## Troubleshooting

### No video capture
- Ensure your webcam is connected and not in use by another application
- Check that OpenCV can access the camera

### Hand not detected
- Ensure good lighting conditions
- Position hand clearly in front of the camera
- MediaPipe requires at least 70% detection confidence

### Map not rendering
- Verify OpenGL/Pygame installation
- Check that graphics drivers are up to date
- Run with administrator privileges if needed

### Zoom not responding
- Ensure the distance between thumb and index finger is visible
- Try spreading fingers more dramatically for zoom in
- Pinch more tightly for zoom out

## Future Enhancements

- [ ] Satellite/terrain map layers
- [ ] Real map data integration (OpenStreetMap)
- [ ] Gestures to change map modes
- [ ] Path drawing with hand traces
- [ ] Keyboard shortcuts for quick controls
- [ ] Map rotation with two-hand gestures
- [ ] Add more map features (markers, layers)
- [ ] Customizable map styles

## Performance Tips

- Use in well-lit environments for better hand tracking
- Avoid reflective objects that might interfere with the camera
- Keep hands within the camera view
- Smooth movements for better gesture recognition

## License

This project uses MediaPipe (Apache 2.0) and PyOpenGL (BSD License).
