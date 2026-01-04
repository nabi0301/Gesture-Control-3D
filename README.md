# Gesture Control 3D Model

A real-time gesture-controlled 3D shape visualization system using hand tracking with MediaPipe and OpenGL.

## Features

- 🎮 **Hand Gesture Recognition**: Real-time hand tracking using MediaPipe
- 🔄 **Multiple 3D Shapes**: Cube, Pyramid, Sphere, and Tetrahedron
- 🎯 **Gesture Nodes**: Visual representation of hand landmarks as 3D nodes
- 🖱️ **Interactive Controls**:
  - **Left Hand**: Rotate 3D model (X/Y axes)
  - **Right Hand Pinch**: Scale/zoom the model
  - **Right Hand Swipe**: Switch between shapes (left/right)

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
- Move UP/DOWN → Rotate X axis
- Move LEFT/RIGHT → Rotate Y axis

**Right Hand:**
- PINCH gesture → Scale/zoom model
- SWIPE RIGHT → Next shape
- SWIPE LEFT → Previous shape

**General:**
- Press 'Q' or close window → Quit

## How It Works

1. **Hand Detection**: Uses MediaPipe's hand tracking to detect hand landmarks in real-time
2. **Gesture Recognition**: Detects hand position and swipe gestures to control the 3D model
3. **3D Rendering**: Uses PyOpenGL to render interactive 3D shapes
4. **Gesture Visualization**: Displays hand landmark positions as colored nodes in 3D space

## Shapes Available

- **Cube**: Colorful 6-sided 3D box
- **Pyramid**: 5-sided pyramid with color-coded faces
- **Sphere**: Orange spherical shape
- **Tetrahedron**: 4-sided triangular shape

## Gesture Node Colors

- 🟦 **Cyan nodes**: Left hand finger tips
- 🟪 **Magenta nodes**: Right hand finger tips
- 🟨 **Yellow node**: Hand palm center

## Performance

- 30 FPS refresh rate
- Real-time hand tracking latency: <50ms
- Optimized for systems with integrated graphics

## Troubleshooting

### No hand detection:
- Ensure good lighting conditions
- Keep hands visible and within frame
- Adjust MediaPipe confidence thresholds in code

### Low performance:
- Reduce window resolution
- Lower the FPS (change `clock.tick(30)` to a lower value)
- Update GPU drivers

## Future Enhancements

- [ ] Additional 3D shapes (cylinder, cone, torus)
- [ ] Color customization for shapes
- [ ] Recording and playback of gestures
- [ ] Multi-hand complex gestures
- [ ] Save/export 3D model states
- [ ] Voice control integration

## Author

Created as a gesture-controlled 3D visualization project.

## License

MIT License

## Contributing

Feel free to fork, modify, and improve this project!
