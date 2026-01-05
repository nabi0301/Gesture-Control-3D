import cv2
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import math
from collections import deque
import requests
from PIL import Image
from io import BytesIO
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
        
        # Real map tile management
        self.latitude = 20.0  # Start at equator
        self.longitude = 0.0  # Start at prime meridian
        self.zoom_level = 3   # Zoom level (1-18)
        self.tile_texture = None
        self.map_dirty = True  # Flag to update map tiles
    
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
    
    def lat_lon_to_tile(self, lat, lon, zoom):
        """Convert latitude/longitude to tile coordinates"""
        n = 2.0 ** zoom
        x = int((lon + 180.0) / 360.0 * n)
        y = int((1.0 - math.log(math.tan(math.radians(lat)) + 1.0 / math.cos(math.radians(lat))) / math.pi) / 2.0 * n)
        return x, y, zoom
    
    def fetch_map_tiles(self, lat, lon, zoom, tile_size=256):
        """Fetch map tiles from OpenStreetMap and composite them"""
        try:
            x, y, z = self.lat_lon_to_tile(lat, lon, zoom)
            
            # Fetch 3x3 grid of tiles around the center
            tiles_data = []
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    tile_x = x + dx
                    tile_y = y + dy
                    
                    # OpenStreetMap tile URL
                    url = f"https://tile.openstreetmap.org/{z}/{tile_x}/{tile_y}.png"
                    
                    try:
                        response = requests.get(url, timeout=2)
                        if response.status_code == 200:
                            img = Image.open(BytesIO(response.content))
                            tiles_data.append(img)
                        else:
                            # Create blank tile if fetch fails
                            tiles_data.append(Image.new('RGB', (tile_size, tile_size), color=(100, 150, 200)))
                    except:
                        # Create blank tile on error
                        tiles_data.append(Image.new('RGB', (tile_size, tile_size), color=(100, 150, 200)))
            
            # Composite 3x3 tiles into one image
            composite = Image.new('RGB', (tile_size * 3, tile_size * 3))
            for i, tile in enumerate(tiles_data):
                row = i // 3
                col = i % 3
                composite.paste(tile, (col * tile_size, row * tile_size))
            
            return composite
        except Exception as e:
            print(f"Error fetching map tiles: {e}")
            # Return a default ocean-colored image
            return Image.new('RGB', (768, 768), color=(50, 120, 200))
    
    def image_to_texture(self, image):
        """Convert PIL image to OpenGL texture"""
        try:
            # Convert image to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image_data = image.tobytes("raw", "RGB", 0, -1)
            
            # Create texture
            texture = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.width, image.height, 0, GL_RGB, GL_UNSIGNED_BYTE, image_data)
            
            return texture
        except Exception as e:
            print(f"Error creating texture: {e}")
            return None
    
    def update_map_tiles(self):
        """Update map tiles based on current position"""
        if self.map_dirty:
            print(f"Fetching map tiles for {self.latitude:.2f}, {self.longitude:.2f} (zoom: {self.zoom_level})")
            map_image = self.fetch_map_tiles(self.latitude, self.longitude, self.zoom_level)
            self.tile_texture = self.image_to_texture(map_image)
            self.map_dirty = False
    
    def detect_pan(self, hand_landmarks, hand_label):
        """Detect hand movement to pan the map"""
        palm_x = hand_landmarks.landmark[9].x
        palm_y = hand_landmarks.landmark[9].y
        
        if hand_label == "Left":
            if self.prev_hand_x is not None and self.prev_hand_y is not None:
                # Calculate delta movement
                delta_x = palm_x - self.prev_hand_x
                delta_y = palm_y - self.prev_hand_y
                
                # Update coordinates based on movement (adjust scale for zoom level)
                scale_factor = 180.0 / (2 ** (self.zoom_level - 1))
                self.longitude -= delta_x * scale_factor
                self.latitude += delta_y * scale_factor
                
                # Keep within valid ranges
                self.longitude = self.longitude % 360
                self.latitude = max(-85.051129, min(85.051129, self.latitude))
                
                self.map_dirty = True
            
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
                zoom_delta = (distance - self.prev_distance) * 10
                if zoom_delta > 0.01:  # Zoom in
                    self.zoom_level = min(18, self.zoom_level + 1)
                    self.map_dirty = True
                elif zoom_delta < -0.01:  # Zoom out
                    self.zoom_level = max(1, self.zoom_level - 1)
                    self.map_dirty = True
            
            self.prev_distance = distance
    
    def draw_world_map(self):
        """Draw real OpenStreetMap tiles as a textured background"""
        # Update tiles if needed
        self.update_map_tiles()
        
        # Draw textured map if tiles are available
        if self.tile_texture is not None:
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, self.tile_texture)
            
            # Draw a large quad with the map texture
            glColor3f(1.0, 1.0, 1.0)
            glBegin(GL_QUADS)
            glTexCoord2f(0, 1)
            glVertex3f(-150, -120, 0)
            glTexCoord2f(1, 1)
            glVertex3f(150, -120, 0)
            glTexCoord2f(1, 0)
            glVertex3f(150, 120, 0)
            glTexCoord2f(0, 0)
            glVertex3f(-150, 120, 0)
            glEnd()
            
            glDisable(GL_TEXTURE_2D)
        else:
            # Fallback: draw ocean blue background
            glColor3f(0.1, 0.4, 0.8)
            glBegin(GL_QUADS)
            glVertex3f(-150, -120, 0)
            glVertex3f(150, -120, 0)
            glVertex3f(150, 120, 0)
            glVertex3f(-150, 120, 0)
            glEnd()
        
        # Draw coordinate info
        glDisable(GL_LIGHTING)
        glColor3f(1.0, 1.0, 1.0)
    
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
        print("Gesture Control - Real Map Navigation")
        print("=" * 50)
        print("\nControls:")
        print("  Left Hand:")
        print("    - Move hand to PAN the map (scroll)")
        print("  Right Hand:")
        print("    - PINCH gesture: ZOOM in/out (1-18 levels)")
        print("    - Move fingers apart: Zoom in")
        print("    - Pinch (bring together): Zoom out")
        print("  Press 'q' or close window to quit")
        print(f"  Starting at: Latitude {self.latitude:.2f}, Longitude {self.longitude:.2f}\n")
        
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
            
            hand_landmarks_list = []
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    hand_label = handedness.classification[0].label
                    hand_landmarks_list.append((hand_landmarks, hand_label))
                    
                    # Detect pan with left hand
                    if hand_label == "Left":
                        self.detect_pan(hand_landmarks, hand_label)
                    
                    # Detect zoom with right hand
                    if hand_label == "Right":
                        self.detect_zoom(hand_landmarks, hand_label)
            
            # Clear screen
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glLoadIdentity()
            
            # Disable depth testing for map background
            glDisable(GL_DEPTH_TEST)
            
            # Draw the map
            self.draw_map()
            
            # Enable depth testing for hand nodes and UI
            glEnable(GL_DEPTH_TEST)
            
            # Draw hand landmark nodes on top
            for hand_landmarks, hand_label in hand_landmarks_list:
                self.draw_gesture_nodes(hand_landmarks, hand_label)
            
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
