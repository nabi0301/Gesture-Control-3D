import cv2
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import math
from collections import deque
try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    import pygame
    from pygame.locals import *
    HAS_OPENGL = True
except ImportError:
    HAS_OPENGL = False

class GestureController3D:
    def __init__(self):
        # Initialize MediaPipe Hand Detection with new API
        self.hands = solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.drawing = solutions.drawing_utils
        
        # Video capture
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # 3D model rotation (in degrees)
        self.rotation_x = 0
        self.rotation_y = 0
        self.rotation_z = 0
        
        # 3D model scale
        self.scale = 1.0
        
        # Gesture detection
        self.prev_distance = 0
        
        # Map management
        self.map_offset_x = 0.0
        self.map_offset_y = 0.0
        self.map_zoom = 1.0
        self.prev_hand_x = None
        self.prev_hand_y = None
        self.map_pan_threshold = 0.05
    
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def detect_gesture(self, hand_landmarks, handedness):
        """Detect hand gestures"""
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        middle_tip = hand_landmarks.landmark[12]
        ring_tip = hand_landmarks.landmark[16]
        pinky_tip = hand_landmarks.landmark[20]
        
        # Calculate thumb-index distance for pinch
        thumb_index_distance = self.calculate_distance(
            (thumb_tip.x, thumb_tip.y),
            (index_tip.x, index_tip.y)
        )
        
        # Check if fingers are closed
        fingers_closed = (
            index_tip.y > hand_landmarks.landmark[6].y and
            middle_tip.y > hand_landmarks.landmark[10].y
        )
        
        if thumb_index_distance < 0.05:
            return "PINCH"
        elif not fingers_closed:
            return "OPEN"
        elif fingers_closed:
            return "FIST"
        else:
            return "NEUTRAL"
    
    def get_hand_position(self, hand_landmarks):
        """Get normalized hand position"""
        palm_x = hand_landmarks.landmark[9].x
        palm_y = hand_landmarks.landmark[9].y
        return np.array([palm_x, palm_y])
    
    def draw_3d_cube(self, frame, rotation_x, rotation_y, rotation_z, scale):
        """Draw a simple 3D cube using 2D projection"""
        h, w, c = frame.shape
        center_x, center_y = w // 2, h // 2
        
        # Define cube vertices
        vertices = np.array([
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1]
        ]) * 50 * scale
        
        # Rotation matrices
        cos_x, sin_x = math.cos(math.radians(rotation_x)), math.sin(math.radians(rotation_x))
        cos_y, sin_y = math.cos(math.radians(rotation_y)), math.sin(math.radians(rotation_y))
        cos_z, sin_z = math.cos(math.radians(rotation_z)), math.sin(math.radians(rotation_z))
        
        # Rotate around X axis
        rot_x = np.array([
            [1, 0, 0],
            [0, cos_x, -sin_x],
            [0, sin_x, cos_x]
        ])
        
        # Rotate around Y axis
        rot_y = np.array([
            [cos_y, 0, sin_y],
            [0, 1, 0],
            [-sin_y, 0, cos_y]
        ])
        
        # Rotate around Z axis
        rot_z = np.array([
            [cos_z, -sin_z, 0],
            [sin_z, cos_z, 0],
            [0, 0, 1]
        ])
        
        # Apply rotations
        rotation_matrix = rot_z @ rot_y @ rot_x
        vertices_rotated = vertices @ rotation_matrix.T
        
        # Project to 2D
        projected = []
        for vertex in vertices_rotated:
            x = vertex[0] + center_x
            y = vertex[1] + center_y
            projected.append([int(x), int(y)])
        
        projected = np.array(projected)
        
        # Draw cube edges
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Front face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Back face
            (0, 4), (1, 5), (2, 6), (3, 7)   # Connecting edges
        ]
        
        for edge in edges:
            pt1 = tuple(projected[edge[0]])
            pt2 = tuple(projected[edge[1]])
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
        
        # Draw vertices
        for point in projected:
            cv2.circle(frame, tuple(point), 5, (255, 0, 0), -1)
        
        return frame
    
    def draw_sphere(self, x, y, z, radius=0.2, color=(1, 1, 1)):
        """Draw a sphere at given position"""
        quad = gluNewQuadric()
        glPushMatrix()
        glTranslatef(x, y, z)
        glColor3f(*color)
        gluSphere(quad, radius, 16, 16)
        glPopMatrix()
    
    def draw_gesture_nodes(self, hand_landmarks, hand_label):
        """Draw hand landmarks as nodes"""
        colors = {
            "Left": (0.2, 0.8, 1.0),   # Cyan for left hand
            "Right": (1.0, 0.2, 0.8)   # Magenta for right hand
        }
        color = colors.get(hand_label, (1.0, 1.0, 1.0))
        
        # Draw nodes for key landmarks (thumb, index, middle, ring, pinky tips)
        key_landmarks = [4, 8, 12, 16, 20]  # Tip positions
        
        for idx in key_landmarks:
            landmark = hand_landmarks.landmark[idx]
            # Convert normalized coordinates to 3D space (-3 to 3)
            x = (landmark.x - 0.5) * 6
            y = -(landmark.y - 0.5) * 6
            z = -3 + landmark.z * 2
            
            self.draw_sphere(x, y, z, radius=0.15, color=color)
        
        # Draw palm center
        palm_x = sum(hand_landmarks.landmark[i].x for i in [0, 5, 9, 13, 17]) / 5
        palm_y = sum(hand_landmarks.landmark[i].y for i in [0, 5, 9, 13, 17]) / 5
        palm_z = sum(hand_landmarks.landmark[i].z for i in [0, 5, 9, 13, 17]) / 5
        
        palm_x = (palm_x - 0.5) * 6
        palm_y = -(palm_y - 0.5) * 6
        palm_z = -3 + palm_z * 2
        
        self.draw_sphere(palm_x, palm_y, palm_z, radius=0.25, color=(1.0, 1.0, 0.0))
    
    def detect_pan(self, hand_landmarks, hand_label):
        """Detect hand movement to pan the map"""
        palm_x = hand_landmarks.landmark[9].x
        palm_y = hand_landmarks.landmark[9].y
        
        if hand_label == "Left":
            if self.prev_hand_x is not None and self.prev_hand_y is not None:
                # Calculate delta movement
                delta_x = palm_x - self.prev_hand_x
                delta_y = palm_y - self.prev_hand_y
                
                # Update map offset based on movement
                self.map_offset_x -= delta_x * 10
                self.map_offset_y += delta_y * 10
            
            self.prev_hand_x = palm_x
            self.prev_hand_y = palm_y
    
    def detect_zoom(self, hand_landmarks, hand_label):
        """Detect pinch gesture to zoom the map"""
        if hand_label == "Right":
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]
            distance = self.calculate_distance(
                (thumb_tip.x, thumb_tip.y),
                (index_tip.x, index_tip.y)
            )
            
            if self.prev_distance > 0:
                # Increase zoom if fingers moving apart, decrease if pinching
                zoom_delta = (distance - self.prev_distance) * 5
                self.map_zoom += zoom_delta
                self.map_zoom = max(0.5, min(3.0, self.map_zoom))
            
            self.prev_distance = distance
    
    def draw_world_map(self):
        """Draw simplified world map with continents and oceans"""
        # Ocean background - fill entire view
        glColor3f(0.1, 0.4, 0.8)  # Ocean blue
        glBegin(GL_QUADS)
        glVertex3f(-150, -120, 0)
        glVertex3f(150, -120, 0)
        glVertex3f(150, 120, 0)
        glVertex3f(-150, 120, 0)
        glEnd()
        
        # Draw continents as large colored rectangles
        continents = [
            # (x1, y1, x2, y2, color) - position on map
            (-120, 20, -60, 70, (0.2, 0.7, 0.2)),    # North America
            (-110, -60, -40, 15, (0.2, 0.7, 0.2)),   # South America
            (-40, 15, 60, 80, (0.25, 0.6, 0.2)),     # Europe + Africa
            (40, -70, 110, 15, (0.2, 0.7, 0.2)),     # Australia area
            (30, 20, 130, 80, (0.22, 0.65, 0.2)),    # Asia
        ]
        
        # Draw continent shapes
        for x1, y1, x2, y2, color in continents:
            glColor3f(*color)
            glBegin(GL_QUADS)
            glVertex3f(x1, y1, 0.01)
            glVertex3f(x2, y1, 0.01)
            glVertex3f(x2, y2, 0.01)
            glVertex3f(x1, y2, 0.01)
            glEnd()
        
        # Draw country/region borders (light lines)
        glColor3f(0.4, 0.4, 0.4)
        glLineWidth(1.0)
        glBegin(GL_LINES)
        
        # Vertical border lines
        for x in range(-120, 140, 30):
            glVertex3f(x, -100, 0.02)
            glVertex3f(x, 100, 0.02)
        
        # Horizontal border lines
        for y in range(-80, 100, 30):
            glVertex3f(-140, y, 0.02)
            glVertex3f(140, y, 0.02)
        
        glEnd()
        glLineWidth(1.0)
        
        # Draw major city markers on continents
        cities = [
            (-90, 45, (1.0, 0.2, 0.2)),      # North America - Red
            (-75, -20, (1.0, 0.2, 0.2)),     # South America - Red
            (10, 50, (1.0, 0.8, 0.0)),       # Europe - Yellow
            (20, 30, (1.0, 0.8, 0.0)),       # Africa - Yellow
            (80, 50, (0.2, 0.8, 1.0)),       # Asia - Cyan
            (75, -25, (0.8, 0.2, 1.0)),      # Australia - Magenta
        ]
        
        # Draw city markers as visible points
        glPointSize(10.0)
        glBegin(GL_POINTS)
        for cx, cy, color in cities:
            glColor3f(*color)
            glVertex3f(cx, cy, 0.05)
        glEnd()
        glPointSize(1.0)
    
    def draw_map_grid(self):
        """Draw an interactive grid-based map"""
        grid_size = 20
        grid_spacing = 2.0
        
        glColor3f(0.3, 0.3, 0.3)
        glBegin(GL_LINES)
        
        # Draw grid lines with offset and zoom
        for i in range(-grid_size, grid_size):
            # Vertical lines
            glVertex3f(i * grid_spacing, -grid_size * grid_spacing, 0)
            glVertex3f(i * grid_spacing, grid_size * grid_spacing, 0)
            
            # Horizontal lines
            glVertex3f(-grid_size * grid_spacing, i * grid_spacing, 0)
            glVertex3f(grid_size * grid_spacing, i * grid_spacing, 0)
        
        glEnd()
        
        # Draw terrain features (cities/landmarks)
        self.draw_map_features()
    
    def draw_map_features(self):
        """Draw map features like cities and roads"""
        # Define some city locations
        cities = [
            (0, 0, "Central"),
            (10, 10, "North East"),
            (-10, 10, "North West"),
            (10, -10, "South East"),
            (-10, -10, "South West"),
            (5, 0, "East"),
            (-5, 0, "West"),
            (0, 5, "North"),
            (0, -5, "South"),
        ]
        
        # Draw cities as colored spheres
        for x, y, name in cities:
            glPushMatrix()
            glTranslatef(x + self.map_offset_x, y + self.map_offset_y, 0.1)
            glColor3f(1.0, 0.5, 0.0)
            quad = gluNewQuadric()
            gluSphere(quad, 0.3 / self.map_zoom, 16, 16)
            glPopMatrix()
        
        # Draw some roads (connecting lines)
        glColor3f(0.5, 0.5, 0.0)
        glBegin(GL_LINES)
        for i in range(len(cities) - 1):
            x1, y1, _ = cities[i]
            x2, y2, _ = cities[i + 1]
            glVertex3f(x1 + self.map_offset_x, y1 + self.map_offset_y, 0.05)
            glVertex3f(x2 + self.map_offset_x, y2 + self.map_offset_y, 0.05)
        glEnd()
        
        # Draw coordinate axes at center
        glLineWidth(2.0)
        glBegin(GL_LINES)
        # X axis (red)
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(-5, 0, 0)
        glVertex3f(5, 0, 0)
        # Y axis (green)
        glColor3f(0.0, 1.0, 0.0)
        glVertex3f(0, -5, 0)
        glVertex3f(0, 5, 0)
        glEnd()
        glLineWidth(1.0)
    
    def draw_map(self):
        """Draw the interactive world map with gesture controls"""
        # Save the current matrix
        glPushMatrix()
        
        # Apply map pan and zoom
        glTranslatef(self.map_offset_x * 0.1, self.map_offset_y * 0.1, 0)
        glScalef(self.map_zoom * 0.8, self.map_zoom * 0.8, 1.0)
        
        # Draw the world map with continents
        self.draw_world_map()
        
        # Restore the matrix
        glPopMatrix()
    
    def draw_opengl_cube(self):
        """Deprecated - replaced with map drawing"""
        pass

    def run_opengl(self):
        """Main loop with OpenGL map visualization"""
        if not HAS_OPENGL:
            print("OpenGL/Pygame not installed. Run: pip install pygame PyOpenGL PyOpenGL_accelerate")
            self.run()  # Fallback to simple mode
            return
        
        pygame.init()
        display = (800, 600)
        screen = pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Gesture Control Map Navigation")
        
        glMatrixMode(GL_PROJECTION)
        gluPerspective(45, (display[0] / display[1]), 0.1, 500.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(0, 0, -150)  # Position camera to view the map
        
        print("=" * 50)
        print("Gesture Control Map Navigation")
        print("=" * 50)
        print("\nControls:")
        print("  Left Hand:")
        print("    - Move hand to PAN the map")
        print("  Right Hand:")
        print("    - PINCH gesture (thumb + index): ZOOM in/out")
        print("    - Move fingers apart: Zoom in")
        print("    - Pinch (bring together): Zoom out")
        print("  Press 'q' or close window to quit\n")
        
        clock = pygame.time.Clock()
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
            
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    hand_label = handedness.classification[0].label
                    
                    # Detect pan with left hand
                    if hand_label == "Left":
                        self.detect_pan(hand_landmarks, hand_label)
                    
                    # Detect zoom with right hand
                    if hand_label == "Right":
                        self.detect_zoom(hand_landmarks, hand_label)
            
            # Disable depth testing for flat map rendering
            glDisable(GL_DEPTH_TEST)
            
            # Clear screen
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glLoadIdentity()
            glTranslatef(0, 0, -150)
            
            # Draw the map
            self.draw_map()
            
            # Re-enable depth testing for 3D elements if needed
            glEnable(GL_DEPTH_TEST)
            
            pygame.display.flip()
            clock.tick(30)
        
        self.cap.release()
        pygame.quit()
        print("\nMap navigation closed successfully!")
    
    def run(self):
        """Main loop"""
        print("=" * 50)
        print("Gesture Control 3D Model")
        print("=" * 50)
        print("\nControls:")
        print("  Left Hand:")
        print("    - Move UP/DOWN: Rotate X axis")
        print("    - Move LEFT/RIGHT: Rotate Y axis")
        print("  Right Hand:")
        print("    - PINCH gesture: Scale model")
        print("  Press 'q' to quit\n")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Flip frame for selfie view
            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape
            
            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    # Draw hand landmarks
                    self.drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        solutions.hands.HAND_CONNECTIONS
                    )
                    
                    hand_label = handedness.classification[0].label  # "Left" or "Right"
                    hand_pos = self.get_hand_position(hand_landmarks)
                    gesture = self.detect_gesture(hand_landmarks, handedness)
                    
                    # Left hand controls rotation
                    if hand_label == "Left":
                        self.rotation_y = (hand_pos[0] - 0.5) * 180
                        self.rotation_x = (hand_pos[1] - 0.5) * 180
                    
                    # Right hand controls scale (via pinch gesture)
                    if hand_label == "Right":
                        thumb_tip = hand_landmarks.landmark[4]
                        index_tip = hand_landmarks.landmark[8]
                        distance = self.calculate_distance(
                            (thumb_tip.x, thumb_tip.y),
                            (index_tip.x, index_tip.y)
                        )
                        
                        if self.prev_distance > 0:
                            self.scale += (distance - self.prev_distance) * 5
                            self.scale = max(0.5, min(2.0, self.scale))
                        
                        self.prev_distance = distance
                    
                    # Display gesture
                    cv2.putText(frame, f"{hand_label}: {gesture}", 
                              (int(hand_pos[0] * w), int(hand_pos[1] * h)),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Draw 3D cube
            frame = self.draw_3d_cube(frame, self.rotation_x, self.rotation_y, self.rotation_z, self.scale)
            
            # Display info
            info_text = f"Rotation: X={self.rotation_x:.1f}° Y={self.rotation_y:.1f}° Z={self.rotation_z:.1f}° | Scale: {self.scale:.2f}"
            # cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show frame (commented out to hide background image)
            # cv2.imshow("Gesture Control 3D Model", frame)
            
            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        print("\nProgram closed successfully!")

if __name__ == "__main__":
    controller = GestureController3D()
    controller.run_opengl()  # Use OpenGL for 3D visualization
