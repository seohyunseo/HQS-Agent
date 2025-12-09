import os
import cv2
import numpy as np
import argparse
from pathlib import Path
import time


class RecordingVisualizer:
    """저장된 RGB-D + Spatial Input 레코딩을 시각화하는 클래스"""

    def __init__(self, session_path):
        self.session_path = Path(session_path)

        # Check paths
        self.rgb_dir = self.session_path / 'rgb'
        self.depth_dir = self.session_path / 'depth'
        self.metadata_file = self.session_path / 'metadata.txt'
        self.si_dir = self.session_path / 'spatial_input'
        self.si_metadata_file = self.si_dir / 'si_metadata.txt' if self.si_dir.exists() else None

        if not self.rgb_dir.exists():
            raise ValueError(f"RGB directory not found: {self.rgb_dir}")
        if not self.depth_dir.exists():
            raise ValueError(f"Depth directory not found: {self.depth_dir}")

        # Load metadata
        self.metadata = self._load_metadata()
        self.si_data = self._load_spatial_input() if self.si_metadata_file and self.si_metadata_file.exists() else {}

        # Get frame list
        self.rgb_files = sorted(list(self.rgb_dir.glob('frame_*.png')))
        self.depth_files = sorted(list(self.depth_dir.glob('frame_*.png')))

        self.total_frames = len(self.rgb_files)

        print(f"[Visualizer] Loaded session: {session_path}")
        print(f"[Visualizer] Total RGB-D frames: {self.total_frames}")
        print(f"[Visualizer] Total SI packets: {len(self.si_data)}")

    def _load_metadata(self):
        """메타데이터 파일 로드"""
        metadata = {}
        min_timestamp = None

        if self.metadata_file.exists():
            # First pass: find minimum timestamp
            with open(self.metadata_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if ',' in line and not line.startswith('Total'):
                        parts = line.split(',')
                        if len(parts) == 2:
                            try:
                                timestamp = float(parts[1])
                                if min_timestamp is None or timestamp < min_timestamp:
                                    min_timestamp = timestamp
                            except:
                                pass

            # Second pass: normalize timestamps
            if min_timestamp is not None:
                with open(self.metadata_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if ',' in line and not line.startswith('Total'):
                            parts = line.split(',')
                            if len(parts) == 2:
                                try:
                                    frame_id = int(parts[0])
                                    # Normalize timestamp to start from 0
                                    timestamp = float(parts[1]) - min_timestamp
                                    metadata[frame_id] = timestamp
                                except:
                                    pass

        if metadata:
            first_ts = metadata[0]
            last_key = max(metadata.keys())
            last_ts = metadata[last_key]
            print(f"[Visualizer] RGB-D time range: {first_ts:.3f}s - {last_ts:.3f}s")

        return metadata

    def _load_spatial_input(self):
        """Spatial Input 데이터 로드"""
        si_data = {}
        min_timestamp = None

        if not self.si_metadata_file.exists():
            return si_data

        # First pass: find minimum timestamp for normalization
        with open(self.si_metadata_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('Total') or line.startswith('Average'):
                    continue
                try:
                    parts = line.split(',')
                    if len(parts) >= 2:
                        timestamp = float(parts[1])
                        if min_timestamp is None or timestamp < min_timestamp:
                            min_timestamp = timestamp
                except:
                    pass

        if min_timestamp is None:
            return si_data

        # Second pass: parse data with normalized timestamps
        with open(self.si_metadata_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('Total') or line.startswith('Average'):
                    continue

                try:
                    parts = line.split(',')
                    if len(parts) < 2:
                        continue

                    packet_id = int(parts[0])
                    # Normalize timestamp to start from 0 and convert to seconds
                    timestamp = (float(parts[1]) - min_timestamp) / 10000000.0  # Convert to seconds

                    si_packet = {
                        'timestamp': timestamp,
                        'head_position': None,
                        'eye_origin': None,
                        'eye_direction': None,
                        'left_hand_position': None,
                        'right_hand_position': None
                    }

                    # Parse data
                    i = 2
                    while i < len(parts):
                        if parts[i] == 'head' and i + 3 < len(parts):
                            si_packet['head_position'] = np.array([
                                float(parts[i + 1]),
                                float(parts[i + 2]),
                                float(parts[i + 3])
                            ])
                            i += 4
                        elif parts[i] == 'eye_origin' and i + 3 < len(parts):
                            si_packet['eye_origin'] = np.array([
                                float(parts[i + 1]),
                                float(parts[i + 2]),
                                float(parts[i + 3])
                            ])
                            i += 4
                        elif parts[i] == 'eye_direction' and i + 3 < len(parts):
                            si_packet['eye_direction'] = np.array([
                                float(parts[i + 1]),
                                float(parts[i + 2]),
                                float(parts[i + 3])
                            ])
                            i += 4
                        elif parts[i] == 'left_hand' and i + 3 < len(parts):
                            si_packet['left_hand_position'] = np.array([
                                float(parts[i + 1]),
                                float(parts[i + 2]),
                                float(parts[i + 3])
                            ])
                            i += 4
                        elif parts[i] == 'right_hand' and i + 3 < len(parts):
                            si_packet['right_hand_position'] = np.array([
                                float(parts[i + 1]),
                                float(parts[i + 2]),
                                float(parts[i + 3])
                            ])
                            i += 4
                        else:
                            i += 1

                    si_data[packet_id] = si_packet

                except Exception as e:
                    # print(f"Error parsing SI line: {line}, {e}")
                    pass

        print(f"[Visualizer] Loaded {len(si_data)} SI packets")
        if si_data:
            first_ts = list(si_data.values())[0]['timestamp']
            last_ts = list(si_data.values())[-1]['timestamp']
            print(f"[Visualizer] SI time range: {first_ts:.3f}s - {last_ts:.3f}s")

        return si_data

    def read_depth(self, depth_path):
        """Depth 이미지 읽기 (uint16 -> float32)"""
        depth_uint16 = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_uint16 is None:
            return None

        # Convert from mm (uint16) to meters (float32)
        depth_meters = depth_uint16.astype(np.float32) / 1000.0
        return depth_meters

    def colorize_depth(self, depth, min_depth=0.0, max_depth=3.0):
        """Depth를 컬러맵으로 시각화"""
        # Clip depth values
        depth_clipped = np.clip(depth, min_depth, max_depth)

        # Normalize to 0-255
        depth_normalized = ((depth_clipped - min_depth) / (max_depth - min_depth) * 255).astype(np.uint8)

        # Apply colormap
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

        # Set invalid depth (0) to black
        mask = depth <= 0
        depth_colored[mask] = [0, 0, 0]

        return depth_colored

    def draw_spatial_input_overlay(self, rgb_img, depth_img, frame_timestamp):
        """RGB 및 Depth 이미지에 Spatial Input 데이터 오버레이"""
        rgb_vis = rgb_img.copy()
        depth_vis = depth_img.copy()

        # Find closest SI packet by timestamp
        if not self.si_data:
            return rgb_vis, depth_vis

        # Find closest timestamp
        closest_packet = None
        min_time_diff = float('inf')

        for packet_id, si_packet in self.si_data.items():
            time_diff = abs(si_packet['timestamp'] - frame_timestamp)
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                closest_packet = si_packet

        # Increased threshold to 1.0 second for better matching
        if closest_packet is None or min_time_diff > 1.0:
            # Show no data message
            cv2.putText(rgb_vis, f"No SI data (diff: {min_time_diff:.3f}s)",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 2)
            cv2.putText(depth_vis, f"No SI data (diff: {min_time_diff:.3f}s)",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 2)
            return rgb_vis, depth_vis

        # Draw on RGB
        rgb_vis = self._draw_si_data(rgb_vis, closest_packet, is_depth=False, time_diff=min_time_diff)

        # Draw on Depth
        depth_vis = self._draw_si_data(depth_vis, closest_packet, is_depth=True, time_diff=min_time_diff)

        return rgb_vis, depth_vis

    def _draw_si_data(self, img, si_packet, is_depth=False, time_diff=0.0):
        """이미지에 Spatial Input 데이터 그리기"""
        h, w = img.shape[:2]

        # Text position
        text_y = 60
        line_height = 25

        # Show time difference
        cv2.putText(img, f"SI sync: {time_diff * 1000:.0f}ms", (10, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        text_y += line_height

        # Head position
        if si_packet['head_position'] is not None:
            head_pos = si_packet['head_position']
            text = f"Head: ({head_pos[0]:.3f}, {head_pos[1]:.3f}, {head_pos[2]:.3f})m"
            cv2.putText(img, text, (10, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 0), 2)
            text_y += line_height

            # Draw head indicator (circle at top center)
            cv2.circle(img, (w // 2, 30), 10, (255, 255, 0), -1)
            cv2.putText(img, "HEAD", (w // 2 - 25, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (255, 255, 0), 1)

        # Eye tracking
        if si_packet['eye_origin'] is not None and si_packet['eye_direction'] is not None:
            eye_origin = si_packet['eye_origin']
            eye_direction = si_packet['eye_direction']
            text = f"Eye Dir: ({eye_direction[0]:.3f}, {eye_direction[1]:.3f}, {eye_direction[2]:.3f})"
            cv2.putText(img, text, (10, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 255), 2)
            text_y += line_height

            # Project eye origin and direction using head as reference
            # Match coordinate system with top-down view

            if si_packet['head_position'] is not None:
                head_pos = si_packet['head_position']

                # Calculate relative position of eye from head
                eye_relative = eye_origin - head_pos

                # Map relative position to pixel offset (inverted X to match top-down)
                offset_scale = 500
                origin_x = int(w / 2 - eye_relative[0] * offset_scale)  # Invert X
                origin_y = int(h / 2 - eye_relative[1] * offset_scale)

                # Clamp origin to image bounds
                origin_x = max(10, min(w - 10, origin_x))
                origin_y = max(10, min(h - 10, origin_y))

                # Draw eye origin as a point
                cv2.circle(img, (origin_x, origin_y), 5, (255, 0, 255), -1)
                cv2.circle(img, (origin_x, origin_y), 8, (255, 0, 255), 2)

                # Project gaze direction
                eye_dir_norm = eye_direction / (np.linalg.norm(eye_direction) + 1e-6)

                if eye_dir_norm[2] > 0.05:  # Looking generally forward
                    # Project direction vector onto image plane
                    # Invert X to match coordinate system
                    focal_length_px = w * 1.0
                    ray_scale = focal_length_px / eye_dir_norm[2]
                    gaze_x = int(origin_x - eye_dir_norm[0] * ray_scale)  # Invert X
                    gaze_y = int(origin_y - eye_dir_norm[1] * ray_scale)

                    # Clamp to image bounds
                    gaze_x = max(5, min(w - 5, gaze_x))
                    gaze_y = max(5, min(h - 5, gaze_y))

                    # Draw gaze ray as arrow from origin
                    cv2.arrowedLine(img, (origin_x, origin_y),
                                    (gaze_x, gaze_y),
                                    (255, 0, 255), 2, tipLength=0.15)

        # Left hand position
        if si_packet['left_hand_position'] is not None:
            left_pos = si_packet['left_hand_position']
            text = f"Left Hand: ({left_pos[0]:.3f}, {left_pos[1]:.3f}, {left_pos[2]:.3f})m"
            cv2.putText(img, text, (10, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)
            text_y += line_height

            # Draw left hand indicator (left side)
            cv2.circle(img, (50, h // 2), 12, (0, 255, 0), -1)
            cv2.putText(img, "L", (45, h // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2)

        # Right hand position
        if si_packet['right_hand_position'] is not None:
            right_pos = si_packet['right_hand_position']
            text = f"Right Hand: ({right_pos[0]:.3f}, {right_pos[1]:.3f}, {right_pos[2]:.3f})m"
            cv2.putText(img, text, (10, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 2)
            text_y += line_height

            # Draw right hand indicator (right side)
            cv2.circle(img, (w - 50, h // 2), 12, (0, 0, 255), -1)
            cv2.putText(img, "R", (w - 55, h // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2)

        return img

    def create_3d_visualization(self, si_packet, size=(400, 400)):
        """3D 공간에서 Spatial Input을 시각화 (Top-down view)"""
        vis = np.zeros((size[0], size[1], 3), dtype=np.uint8)

        if si_packet is None:
            cv2.putText(vis, "No SI Data", (size[0] // 4, size[1] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            return vis

        # Draw grid
        center_x, center_y = size[0] // 2, size[1] // 2
        scale = 100  # pixels per meter

        # Grid lines (every 0.5m)
        for i in range(-2, 3):
            y = center_y + int(i * 0.5 * scale)
            if 0 <= y < size[0]:
                cv2.line(vis, (0, y), (size[1], y), (50, 50, 50), 1)
            x = center_x + int(i * 0.5 * scale)
            if 0 <= x < size[1]:
                cv2.line(vis, (x, 0), (x, size[0]), (50, 50, 50), 1)

        # Draw axes
        cv2.line(vis, (center_x, 0), (center_x, size[0]), (100, 100, 100), 2)  # Z axis
        cv2.line(vis, (0, center_y), (size[1], center_y), (100, 100, 100), 2)  # X axis

        # Labels
        cv2.putText(vis, "X+", (size[1] - 30, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (150, 150, 150), 1)
        cv2.putText(vis, "Z+", (center_x + 10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (150, 150, 150), 1)

        # Draw head (if available)
        if si_packet['head_position'] is not None:
            head_pos = si_packet['head_position']
            # Top-down: X -> horizontal (inverted for correct orientation), Z -> vertical (inverted)
            px = center_x - int(head_pos[0] * scale)
            py = center_y - int(head_pos[2] * scale)  # Z is forward

            if 0 <= px < size[1] and 0 <= py < size[0]:
                cv2.circle(vis, (px, py), 15, (255, 255, 0), -1)
                cv2.putText(vis, "HEAD", (px - 20, py - 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (255, 255, 0), 1)

        # Draw left hand
        if si_packet['left_hand_position'] is not None:
            left_pos = si_packet['left_hand_position']
            px = center_x - int(left_pos[0] * scale)  # Invert X for correct left/right
            py = center_y - int(left_pos[2] * scale)

            if 0 <= px < size[1] and 0 <= py < size[0]:
                cv2.circle(vis, (px, py), 12, (0, 255, 0), -1)
                cv2.circle(vis, (px, py), 12, (255, 255, 255), 2)
                cv2.putText(vis, "L", (px - 5, py + 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 2)

        # Draw right hand
        if si_packet['right_hand_position'] is not None:
            right_pos = si_packet['right_hand_position']
            px = center_x - int(right_pos[0] * scale)  # Invert X for correct left/right
            py = center_y - int(right_pos[2] * scale)

            if 0 <= px < size[1] and 0 <= py < size[0]:
                cv2.circle(vis, (px, py), 12, (0, 0, 255), -1)
                cv2.circle(vis, (px, py), 12, (255, 255, 255), 2)
                cv2.putText(vis, "R", (px - 5, py + 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 2)

        # Draw eye gaze ray (if available)
        if si_packet['eye_origin'] is not None and si_packet['eye_direction'] is not None:
            eye_origin = si_packet['eye_origin']
            eye_dir = si_packet['eye_direction']

            origin_x = center_x - int(eye_origin[0] * scale)
            origin_y = center_y - int(eye_origin[2] * scale)

            gaze_length = 1.0
            end_x = origin_x - int(eye_dir[0] * gaze_length * scale)
            end_y = origin_y - int(eye_dir[2] * gaze_length * scale)

            if 0 <= origin_x < size[1] and 0 <= origin_y < size[0]:
                cv2.arrowedLine(vis, (origin_x, origin_y), (end_x, end_y),
                                (255, 0, 255), 2, tipLength=0.2)
                cv2.circle(vis, (origin_x, origin_y), 5, (255, 0, 255), -1)

        # Title
        cv2.putText(vis, "Top-Down View", (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200, 200, 200), 2)
        cv2.putText(vis, "(1 grid = 0.5m)", (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (150, 150, 150), 1)

        return vis

    def visualize_frame(self, frame_idx, min_depth=0.0, max_depth=3.0):
        """특정 프레임 시각화"""
        if frame_idx < 0 or frame_idx >= self.total_frames:
            return None, None, None

        # Read RGB
        rgb = cv2.imread(str(self.rgb_files[frame_idx]))

        # Read and colorize depth
        depth = self.read_depth(self.depth_files[frame_idx])
        depth_colored = self.colorize_depth(depth, min_depth, max_depth) if depth is not None else None

        # Get frame timestamp
        frame_timestamp = self.metadata.get(frame_idx, 0.0)

        # Overlay spatial input data
        if depth_colored is not None:
            rgb_vis, depth_vis = self.draw_spatial_input_overlay(rgb, depth_colored, frame_timestamp)
        else:
            rgb_vis, depth_vis = rgb, depth_colored

        # Find closest SI packet for 3D visualization
        closest_packet = None
        if self.si_data:
            min_time_diff = float('inf')
            for packet_id, si_packet in self.si_data.items():
                time_diff = abs(si_packet['timestamp'] - frame_timestamp)
                if time_diff < min_time_diff and time_diff < 1.0:  # 1 second threshold
                    min_time_diff = time_diff
                    closest_packet = si_packet

        # Create 3D visualization
        si_3d_vis = self.create_3d_visualization(closest_packet)

        return rgb_vis, depth_vis, si_3d_vis

    def play(self, fps=30, min_depth=0.0, max_depth=3.0, start_frame=0):
        """레코딩 재생"""
        print(f"\n[Visualizer] Playing recording at {fps} FPS")
        print("Controls:")
        print("  - Space: Pause/Resume")
        print("  - Left/Right Arrow: Previous/Next frame (when paused)")
        print("  - 'r': Reset to first frame")
        print("  - '+/-': Increase/Decrease max depth range")
        print("  - 'q': Quit\n")

        cv2.namedWindow('RGB', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Depth', cv2.WINDOW_NORMAL)
        cv2.namedWindow('3D View', cv2.WINDOW_NORMAL)

        # Resize windows
        cv2.resizeWindow('RGB', 640, 360)
        cv2.resizeWindow('Depth', 640, 360)
        cv2.resizeWindow('3D View', 400, 400)

        frame_idx = start_frame
        paused = False
        frame_delay = int(1000 / fps)  # ms

        while True:
            if not paused:
                # Read and display frame
                rgb, depth_colored, si_3d = self.visualize_frame(frame_idx, min_depth, max_depth)

                if rgb is None or depth_colored is None:
                    print(f"[Visualizer] Error reading frame {frame_idx}")
                    break

                # Add frame info to RGB
                timestamp = self.metadata.get(frame_idx, 0.0)
                info_text = f"Frame: {frame_idx}/{self.total_frames - 1} | Time: {timestamp:.3f}s | Depth: {min_depth:.1f}-{max_depth:.1f}m"
                cv2.putText(rgb, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)

                # Show images
                cv2.imshow('RGB', rgb)
                cv2.imshow('Depth', depth_colored)
                cv2.imshow('3D View', si_3d)

                frame_idx += 1
                if frame_idx >= self.total_frames:
                    print("[Visualizer] Reached end of recording")
                    frame_idx = 0  # Loop

            # Handle keyboard input
            key = cv2.waitKey(frame_delay if not paused else 50)

            if key == ord('q'):
                print("[Visualizer] Quit")
                break
            elif key == ord(' '):  # Space - pause/resume
                paused = not paused
                status = "Paused" if paused else "Playing"
                print(f"[Visualizer] {status} at frame {frame_idx}")
            elif key == ord('r'):  # Reset
                frame_idx = 0
                print("[Visualizer] Reset to frame 0")
            elif key == 83 and paused:  # Right arrow - next frame (when paused)
                frame_idx = min(frame_idx + 1, self.total_frames - 1)
            elif key == 81 and paused:  # Left arrow - previous frame (when paused)
                frame_idx = max(frame_idx - 1, 0)
            elif key == ord('+') or key == ord('='):  # Increase max depth
                max_depth += 0.5
                print(f"[Visualizer] Max depth: {max_depth:.1f}m")
            elif key == ord('-'):  # Decrease max depth
                max_depth = max(0.5, max_depth - 0.5)
                print(f"[Visualizer] Max depth: {max_depth:.1f}m")

        cv2.destroyAllWindows()

    def export_video(self, output_path, fps=30, min_depth=0.0, max_depth=3.0):
        """비디오로 내보내기 (RGB, Depth, 3D View를 가로로 배치)"""
        print(f"[Visualizer] Exporting to video: {output_path}")

        # Get first frame to determine size
        rgb, depth_colored, si_3d = self.visualize_frame(0, min_depth, max_depth)
        if rgb is None:
            print("[Visualizer] Error: Cannot read first frame")
            return

        h_rgb, w_rgb = rgb.shape[:2]
        h_3d, w_3d = si_3d.shape[:2]

        # Resize 3D view to match RGB height
        si_3d_resized = cv2.resize(si_3d, (int(w_3d * h_rgb / h_3d), h_rgb))

        # Create combined frame (RGB + Depth + 3D View)
        combined_width = w_rgb * 2 + si_3d_resized.shape[1]

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (combined_width, h_rgb))

        for frame_idx in range(self.total_frames):
            rgb, depth_colored, si_3d = self.visualize_frame(frame_idx, min_depth, max_depth)

            if rgb is None or depth_colored is None:
                print(f"[Visualizer] Warning: Skipping frame {frame_idx}")
                continue

            # Resize 3D view
            si_3d_resized = cv2.resize(si_3d, (int(w_3d * h_rgb / h_3d), h_rgb))

            # Create combined view
            combined = np.hstack([rgb, depth_colored, si_3d_resized])

            # Add frame info
            timestamp = self.metadata.get(frame_idx, 0.0)
            info_text = f"Frame: {frame_idx}/{self.total_frames - 1} | Time: {timestamp:.3f}s"
            cv2.putText(combined, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

            out.write(combined)

            # Progress
            if (frame_idx + 1) % 30 == 0:
                progress = (frame_idx + 1) / self.total_frames * 100
                print(f"[Visualizer] Progress: {progress:.1f}% ({frame_idx + 1}/{self.total_frames})")

        out.release()
        print(f"[Visualizer] Video saved: {output_path}")

    def extract_frames(self, output_dir, frame_indices=None, min_depth=0.0, max_depth=3.0):
        """특정 프레임들을 이미지로 추출"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if frame_indices is None:
            frame_indices = range(self.total_frames)

        print(f"[Visualizer] Extracting {len(frame_indices)} frames to {output_dir}")

        for frame_idx in frame_indices:
            rgb, depth_colored, si_3d = self.visualize_frame(frame_idx, min_depth, max_depth)

            if rgb is None or depth_colored is None:
                continue

            # Save RGB
            rgb_path = output_dir / f'frame_{frame_idx:06d}_rgb.png'
            cv2.imwrite(str(rgb_path), rgb)

            # Save depth
            depth_path = output_dir / f'frame_{frame_idx:06d}_depth.png'
            cv2.imwrite(str(depth_path), depth_colored)

            # Save 3D view
            si_3d_path = output_dir / f'frame_{frame_idx:06d}_3d.png'
            cv2.imwrite(str(si_3d_path), si_3d)

        print(f"[Visualizer] Extraction complete")


def main():
    parser = argparse.ArgumentParser(description='Visualize HoloLens2 RGB-D recordings with Spatial Input')
    parser.add_argument('-session_path', type=str, default='recordings/20251123_163544',
                        help='Path to recording session directory (e.g., recordings/20231123_143025)')
    parser.add_argument('--mode', type=str, default='play', choices=['play', 'video', 'extract'],
                        help='Visualization mode: play (interactive), video (export), extract (save frames)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Playback FPS (default: 30)')
    parser.add_argument('--min-depth', type=float, default=0.0,
                        help='Minimum depth for visualization (meters, default: 0.0)')
    parser.add_argument('--max-depth', type=float, default=3.0,
                        help='Maximum depth for visualization (meters, default: 3.0)')
    parser.add_argument('--output', type=str, default='output.mp4',
                        help='Output path for video mode (default: output.mp4)')
    parser.add_argument('--output-dir', type=str, default='extracted_frames',
                        help='Output directory for extract mode (default: extracted_frames)')
    parser.add_argument('--start-frame', type=int, default=0,
                        help='Start frame index (default: 0)')

    args = parser.parse_args()

    # Create visualizer
    try:
        visualizer = RecordingVisualizer(args.session_path)
    except Exception as e:
        print(f"Error loading session: {e}")
        return

    # Run visualization
    if args.mode == 'play':
        visualizer.play(fps=args.fps, min_depth=args.min_depth, max_depth=args.max_depth,
                        start_frame=args.start_frame)
    elif args.mode == 'video':
        visualizer.export_video(args.output, fps=args.fps,
                                min_depth=args.min_depth, max_depth=args.max_depth)
    elif args.mode == 'extract':
        # Extract every 10th frame by default
        frame_indices = range(0, visualizer.total_frames, 10)
        visualizer.extract_frames(args.output_dir, frame_indices=frame_indices,
                                  min_depth=args.min_depth, max_depth=args.max_depth)


if __name__ == '__main__':
    main()