import sys, os
import asyncio
import time
from collections import deque
import cv2
import numpy as np
import struct
import threading
from queue import Queue
from datetime import datetime

sys.path.append("./hl2ss_")
import hl2ss
import hl2ss_lnm
import hl2ss_mp
import hl2ss_3dcv
import hl2ss_utilities
import socket
import multiprocessing as mp
import keyboard

## Set HoloLens2 wifi address ##
host = '192.168.0.55'

# Calibration path (must exist but can be empty)
calibration_path = 'calibration'

# Front RGB camera parameters
pv_width = 640  # (1080, 1920), (720, 1280), (360, 640), (240, 424)
pv_height = 360
pv_fps = 30

# Buffer length in seconds
buffer_size = 10

# Save settings
flag_depth = True  # Set to False if you don't want to save depth
save_dir = 'recordings'
max_queue_size = 100  # Maximum number of frames in save queue

hand_joint = hl2ss.SI_HandJointKind.Wrist
hand_join_name = hl2ss.si_get_joint_name(hand_joint)


class SpatialInputThread:
    """90fps로 동작하는 Spatial Input을 처리하는 별도 스레드"""

    def __init__(self, host, save_dir, session_name):
        self.host = host
        self.running = False
        self.thread = None
        self.client = None

        # Save directory for spatial input data
        self.si_dir = os.path.join(save_dir, session_name, 'spatial_input')
        os.makedirs(self.si_dir, exist_ok=True)

        # Metadata file
        self.si_metadata_file = os.path.join(self.si_dir, 'si_metadata.txt')

        # Statistics
        self.packet_count = 0
        self.start_time = None

        # Queue for saving (optional)
        self.save_queue = Queue(maxsize=1000)
        self.save_thread = None

    def start(self):
        """Spatial Input 스레드 시작"""
        self.running = True
        self.thread = threading.Thread(target=self._si_worker, daemon=True)
        self.thread.start()

        # Start save worker thread
        self.save_thread = threading.Thread(target=self._save_worker, daemon=True)
        self.save_thread.start()

        print("[SpatialInputThread] Started")

    def stop(self):
        """Spatial Input 스레드 종료"""
        if self.running:
            self.running = False

            # Signal save thread to stop
            self.save_queue.put(None)

            if self.thread:
                self.thread.join(timeout=2.0)
            if self.save_thread:
                self.save_thread.join(timeout=2.0)

            if self.client:
                try:
                    self.client.close()
                except:
                    pass

            print(f"[SpatialInputThread] Stopped. Total packets: {self.packet_count}")

            # Write final metadata
            with open(self.si_metadata_file, 'a') as f:
                f.write(f"\nTotal packets: {self.packet_count}\n")
                if self.start_time:
                    elapsed = time.time() - self.start_time
                    avg_fps = self.packet_count / elapsed if elapsed > 0 else 0
                    f.write(f"Average FPS: {avg_fps:.2f}\n")

    def _si_worker(self):
        """Spatial Input 데이터를 90fps로 수신하는 워커 스레드"""
        print("[SpatialInputThread] Worker started")

        try:
            # Open Spatial Input client
            self.client = hl2ss_lnm.rx_si(self.host, hl2ss.StreamPort.SPATIAL_INPUT)
            self.client.open()

            self.start_time = time.time()
            last_print_time = self.start_time

            while self.running:
                try:
                    # Get next spatial input packet (90fps)
                    data = self.client.get_next_packet()
                    si = data.payload
                    timestamp = data.timestamp

                    # Parse spatial input data
                    si_data = {
                        'timestamp': timestamp,
                        'packet_id': self.packet_count,
                        'head_pose_valid': si.head_pose_valid,
                        'eye_ray_valid': si.eye_ray_valid,
                        'hand_left_valid': si.hand_left_valid,
                        'hand_right_valid': si.hand_right_valid,
                    }

                    # Head pose
                    if si.head_pose_valid:
                        head_pose = si.head_pose
                        si_data['head_position'] = head_pose.position
                        si_data['head_forward'] = head_pose.forward
                        si_data['head_up'] = head_pose.up

                    # Eye tracking
                    if si.eye_ray_valid:
                        eye_ray = si.eye_ray
                        si_data['eye_origin'] = eye_ray.origin
                        si_data['eye_direction'] = eye_ray.direction

                    # Left hand
                    if si.hand_left_valid:
                        hand = si.hand_left
                        si_data['left_hand_orientation'] = hand.orientation[hand_joint, :]
                        si_data['left_hand_position'] = hand.position[hand_joint, :]
                        si_data['left_hand_radius'] = hand.radius[hand_joint]
                        si_data['left_hand_accuracy'] = hand.accuracy[hand_joint]

                    # Right hand
                    if si.hand_right_valid:
                        hand = si.hand_right
                        si_data['right_hand_orientation'] = hand.orientation[hand_joint, :]
                        si_data['right_hand_position'] = hand.position[hand_joint, :]
                        si_data['right_hand_radius'] = hand.radius[hand_joint]
                        si_data['right_hand_accuracy'] = hand.accuracy[hand_joint]

                    # Add to save queue
                    if not self.save_queue.full():
                        self.save_queue.put_nowait(si_data)

                    self.packet_count += 1

                    # Print status every second
                    current_time = time.time()
                    if current_time - last_print_time >= 1.0:
                        elapsed = current_time - self.start_time
                        fps = self.packet_count / elapsed if elapsed > 0 else 0
                        queue_size = self.save_queue.qsize()
                        print(
                            f"[SpatialInputThread] Packets: {self.packet_count} | FPS: {fps:.1f} | Remain Queue: {queue_size}")
                        last_print_time = current_time

                except Exception as e:
                    if self.running:
                        print(f"[SpatialInputThread] Error receiving packet: {e}")

        except Exception as e:
            print(f"[SpatialInputThread] Error opening client: {e}")

        print("[SpatialInputThread] Worker stopped")

    def _save_worker(self):
        """Spatial Input 데이터를 저장하는 워커 스레드"""
        print("[SpatialInputThread] Save worker started")

        while self.running:
            try:
                si_data = self.save_queue.get(timeout=1.0)

                if si_data is None:
                    break

                # Save to text file (you can change to binary format for efficiency)
                with open(self.si_metadata_file, 'a') as f:
                    packet_id = si_data['packet_id']
                    timestamp = si_data['timestamp']

                    line = f"{packet_id},{timestamp}"

                    # Add head pose
                    if si_data['head_pose_valid']:
                        head_pos = si_data['head_position']
                        line += f",head,{head_pos[0]:.6f},{head_pos[1]:.6f},{head_pos[2]:.6f}"

                    # Add eye tracking
                    if si_data['eye_ray_valid']:
                        eye_origin = si_data['eye_origin']
                        eye_direction = si_data['eye_direction']
                        line += f",eye_origin,{eye_origin[0]:.6f},{eye_origin[1]:.6f},{eye_origin[2]:.6f}"
                        line += f",eye_direction,{eye_direction[0]:.6f},{eye_direction[1]:.6f},{eye_direction[2]:.6f}"

                    # Add left hand
                    if si_data['hand_left_valid']:
                        left_pos = si_data['left_hand_position']
                        line += f",left_hand,{left_pos[0]:.6f},{left_pos[1]:.6f},{left_pos[2]:.6f}"

                    # Add right hand
                    if si_data['hand_right_valid']:
                        right_pos = si_data['right_hand_position']
                        line += f",right_hand,{right_pos[0]:.6f},{right_pos[1]:.6f},{right_pos[2]:.6f}"

                    f.write(line + "\n")

                self.save_queue.task_done()

            except Exception as e:
                if self.running:
                    if "Empty" not in str(e):
                        print(f"[SpatialInputThread] Error in save worker: {e}")

        print("[SpatialInputThread] Save worker stopped")


class AudioThread:
    """마이크 오디오를 처리하는 별도 스레드"""

    def __init__(self, host, save_dir, session_name):
        self.host = host
        self.running = False
        self.thread = None
        self.client = None

        # Save directory for audio data
        self.audio_dir = os.path.join(save_dir, session_name, 'audio')
        os.makedirs(self.audio_dir, exist_ok=True)

        # Audio file path
        self.audio_file = os.path.join(self.audio_dir, 'audio.wav')
        self.metadata_file = os.path.join(self.audio_dir, 'audio_metadata.txt')

        # Audio settings
        self.profile = hl2ss.AudioProfile.AAC_24000
        self.channels = hl2ss.Parameters_MICROPHONE.CHANNELS
        self.sample_rate = hl2ss.Parameters_MICROPHONE.SAMPLE_RATE

        # Override sample rate if needed
        # If audio plays too slow, the actual data rate is higher than what we're saving
        # Common rates: 16000, 24000, 48000
        self.output_sample_rate = None  # Will be auto-detected

        # Statistics
        self.packet_count = 0
        self.start_time = None

        # Queue for saving
        self.save_queue = Queue(maxsize=500)
        self.save_thread = None

        # Audio samples buffer
        self.audio_samples = []

    def start(self):
        """오디오 스레드 시작"""
        self.running = True
        self.thread = threading.Thread(target=self._audio_worker, daemon=True)
        self.thread.start()

        # Start save worker thread
        self.save_thread = threading.Thread(target=self._save_worker, daemon=True)
        self.save_thread.start()

        print("[AudioThread] Started")

    def stop(self):
        """오디오 스레드 종료"""
        if self.running:
            self.running = False

            # Signal save thread to stop
            self.save_queue.put(None)

            if self.thread:
                self.thread.join(timeout=2.0)
            if self.save_thread:
                self.save_thread.join(timeout=5.0)

            if self.client:
                try:
                    self.client.close()
                except:
                    pass

            print(f"[AudioThread] Stopped. Total packets: {self.packet_count}")

            # Write final metadata
            with open(self.metadata_file, 'a') as f:
                f.write(f"\nTotal packets: {self.packet_count}\n")
                if self.start_time:
                    elapsed = time.time() - self.start_time
                    f.write(f"Duration: {elapsed:.2f}s\n")
                f.write(f"Sample rate: {self.sample_rate}Hz\n")
                f.write(f"Channels: {self.channels}\n")

    def _audio_worker(self):
        """오디오 데이터를 수신하는 워커 스레드"""
        print("[AudioThread] Worker started")

        try:
            # Open microphone client
            self.client = hl2ss_lnm.rx_microphone(self.host, hl2ss.StreamPort.MICROPHONE,
                                                  profile=self.profile)
            self.client.open()

            self.start_time = time.time()
            last_print_time = self.start_time

            # Flag to print first packet info
            first_packet = True

            while self.running:
                try:
                    # Get next audio packet
                    data = self.client.get_next_packet()
                    timestamp = data.timestamp

                    # Extract audio samples
                    audio_data = data.payload

                    # Print first packet info for debugging
                    if first_packet:
                        print(f"[AudioThread] First packet info:")
                        print(f"  Type: {type(audio_data)}")
                        if isinstance(audio_data, np.ndarray):
                            print(f"  Shape: {audio_data.shape}")
                            print(f"  Dtype: {audio_data.dtype}")
                            print(f"  Min: {audio_data.min():.6f}, Max: {audio_data.max():.6f}")
                        print(f"  Expected: {self.channels} channels, {self.sample_rate}Hz")
                        first_packet = False

                    # Add to save queue
                    if not self.save_queue.full():
                        self.save_queue.put_nowait({
                            'timestamp': timestamp,
                            'packet_id': self.packet_count,
                            'audio_data': audio_data
                        })

                    self.packet_count += 1

                    # Print status every second
                    current_time = time.time()
                    if current_time - last_print_time >= 1.0:
                        elapsed = current_time - self.start_time
                        fps = self.packet_count / elapsed if elapsed > 0 else 0
                        queue_size = self.save_queue.qsize()
                        print(f"[AudioThread] Packets: {self.packet_count} | Rate: {fps:.1f}pps | Queue: {queue_size}")
                        last_print_time = current_time

                except Exception as e:
                    if self.running:
                        print(f"[AudioThread] Error receiving packet: {e}")

        except Exception as e:
            print(f"[AudioThread] Error opening client: {e}")

        print("[AudioThread] Worker stopped")

    def _save_worker(self):
        """오디오 데이터를 저장하는 워커 스레드"""
        print("[AudioThread] Save worker started")

        # Also save raw metadata for debugging
        raw_metadata = []

        while self.running:
            try:
                audio_packet = self.save_queue.get(timeout=1.0)

                if audio_packet is None:
                    break

                # Accumulate audio samples
                audio_data = audio_packet['audio_data']

                # Store metadata for first packet
                if len(self.audio_samples) == 0:
                    raw_metadata.append(f"First packet type: {type(audio_data)}")
                    if isinstance(audio_data, np.ndarray):
                        raw_metadata.append(f"Shape: {audio_data.shape}")
                        raw_metadata.append(f"Dtype: {audio_data.dtype}")

                # Convert to numpy array if needed
                if isinstance(audio_data, np.ndarray):
                    self.audio_samples.append(audio_data)
                elif hasattr(audio_data, 'tobytes'):
                    self.audio_samples.append(audio_data)
                else:
                    # Try to convert
                    self.audio_samples.append(np.frombuffer(audio_data, dtype=np.float32))

                self.save_queue.task_done()

            except Exception as e:
                if self.running:
                    if "Empty" not in str(e):
                        print(f"[AudioThread] Error in save worker: {e}")

        # Save raw metadata
        if raw_metadata:
            with open(self.metadata_file, 'w') as f:
                f.write("\n".join(raw_metadata) + "\n")

        # Save accumulated audio to WAV file
        print("[AudioThread] Saving audio to WAV file...")
        self._save_wav()
        print("[AudioThread] Save worker stopped")

    def _save_wav(self):
        """WAV 파일로 저장"""
        if not self.audio_samples:
            print("[AudioThread] No audio samples to save")
            return

        try:
            import wave

            # Concatenate all audio samples
            all_samples = np.concatenate(self.audio_samples)

            print(f"[AudioThread] Concatenated shape: {all_samples.shape}, dtype: {all_samples.dtype}")

            # Handle planar format (AAC decodes to planar float32)
            # Use hl2ss function to convert planar to packed (interleaved)
            if len(all_samples.shape) == 2 and all_samples.shape[0] == self.channels:
                # Planar format detected: (channels, samples)
                print(f"[AudioThread] Converting from planar to packed format using hl2ss")
                all_samples = hl2ss.microphone_planar_to_packed(all_samples, self.channels)

            # Convert float32 to int16 for WAV
            if all_samples.dtype == np.float32 or all_samples.dtype == np.float64:
                # Clip to [-1, 1] range first
                all_samples = np.clip(all_samples, -1.0, 1.0)
                # Convert to int16
                all_samples = (all_samples * 32767).astype(np.int16)
            elif all_samples.dtype != np.int16:
                # Convert to int16
                all_samples = all_samples.astype(np.int16)

            # Use 24kHz
            self.actual_sample_rate = 24000
            print(f"[AudioThread] Using 24000Hz")

            num_frames = len(all_samples) // self.channels

            # Save as WAV
            with wave.open(self.audio_file, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.actual_sample_rate)
                wav_file.writeframes(all_samples.tobytes())

            # Duration already calculated
            duration = num_frames / self.actual_sample_rate
            duration = num_frames / self.actual_sample_rate

            print(f"[AudioThread] Saved audio: {self.audio_file}")
            print(f"[AudioThread] Sample rate: {self.actual_sample_rate}Hz, Channels: {self.channels}")
            print(f"[AudioThread] Total samples: {len(all_samples)}, Frames: {num_frames}, Duration: {duration:.2f}s")

            # Save detailed metadata
            with open(self.metadata_file, 'a') as f:
                f.write(f"Actual sample rate: {self.actual_sample_rate}Hz\n")
                f.write(f"Total samples: {len(all_samples)}\n")
                f.write(f"Frames: {num_frames}\n")
                f.write(f"Duration: {duration:.2f}s\n")

        except Exception as e:
            print(f"[AudioThread] Error saving WAV: {e}")
            import traceback
            traceback.print_exc()


class AsyncSaver:
    """비동기 이미지 저장을 위한 클래스"""

    def __init__(self, save_dir, max_queue_size=100):
        self.save_dir = save_dir
        self.save_queue = Queue(maxsize=max_queue_size)
        self.running = False
        self.thread = None
        self.frame_count = 0
        self.session_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create save directories
        self.rgb_dir = os.path.join(save_dir, self.session_name, 'rgb')
        self.depth_dir = os.path.join(save_dir, self.session_name, 'depth')
        os.makedirs(self.rgb_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)

        # Metadata file
        self.metadata_file = os.path.join(save_dir, self.session_name, 'metadata.txt')

    def start(self):
        """저장 스레드 시작"""
        self.running = True
        self.thread = threading.Thread(target=self._save_worker, daemon=True)
        self.thread.start()
        print(f"[AsyncSaver] Started. Saving to: {os.path.join(self.save_dir, self.session_name)}")

    def stop(self):
        """저장 스레드 종료"""
        if self.running:
            self.running = False
            # Sentinel value to signal thread to stop
            self.save_queue.put(None)
            if self.thread:
                self.thread.join(timeout=5.0)
            print(f"[AsyncSaver] Stopped. Total frames saved: {self.frame_count}")

            # Write final metadata
            with open(self.metadata_file, 'a') as f:
                f.write(f"\nTotal frames: {self.frame_count}\n")

    def add_frame(self, color, depth, timestamp):
        """프레임을 저장 큐에 추가"""
        try:
            # 큐가 가득 차면 가장 오래된 프레임을 버림 (논블로킹)
            if self.save_queue.full():
                try:
                    self.save_queue.get_nowait()
                    print(f"[AsyncSaver] Warning: Queue full, dropped oldest frame")
                except:
                    pass

            self.save_queue.put_nowait({
                'color': color.copy(),  # Copy to avoid race condition
                'depth': depth.copy() if depth is not None else None,
                'timestamp': timestamp,
                'frame_id': self.frame_count
            })
            self.frame_count += 1

        except Exception as e:
            print(f"[AsyncSaver] Error adding frame: {e}")

    def _save_worker(self):
        """백그라운드에서 이미지를 저장하는 워커 스레드"""
        print("[AsyncSaver] Worker thread started")

        while self.running:
            try:
                # Get frame from queue (blocking with timeout)
                frame_data = self.save_queue.get(timeout=1.0)

                # Check for sentinel value
                if frame_data is None:
                    break

                frame_id = frame_data['frame_id']
                timestamp = frame_data['timestamp']

                # Save RGB
                rgb_path = os.path.join(self.rgb_dir, f'frame_{timestamp}.png')
                cv2.imwrite(rgb_path, frame_data['color'])

                # Save Depth
                if frame_data['depth'] is not None:
                    depth_path = os.path.join(self.depth_dir, f'frame_{timestamp}.png')
                    # Scale depth to uint16 for better precision
                    depth_scaled = (frame_data['depth'] * 1000).astype(np.uint16)  # mm 단위
                    cv2.imwrite(depth_path, depth_scaled)

                # Write metadata
                with open(self.metadata_file, 'a') as f:
                    f.write(f"{frame_id},{timestamp}\n")

                self.save_queue.task_done()

            except Exception as e:
                if self.running:  # Only print if not shutting down
                    if "Empty" not in str(e):
                        print(f"[AsyncSaver] Error in worker: {e}")

        print("[AsyncSaver] Worker thread stopped")

    def get_queue_size(self):
        """현재 큐에 대기 중인 프레임 수 반환"""
        return self.save_queue.qsize()

    def get_session_name(self):
        """세션 이름 반환"""
        return self.session_name


def main():
    ###################### init comm. with hololens2 ######################
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    init_variables, max_depth, producer = init_hl2()

    # Initialize async saver
    saver = AsyncSaver(save_dir, max_queue_size)
    saver.start()

    # Initialize spatial input thread (90fps)
    si_thread = SpatialInputThread(host, save_dir, saver.get_session_name())
    si_thread.start()

    # Initialize audio thread
    audio_thread = AudioThread(host, save_dir, saver.get_session_name())
    audio_thread.start()

    cv2.namedWindow('Prompt')
    cv2.resizeWindow(winname='Prompt', width=500, height=500)
    cv2.moveWindow(winname='Prompt', x=2000, y=200)

    print("\n=== Recording Started ===")
    print("Press 'q' to stop recording\n")

    try:
        frame_count = 0
        start_time = time.time()

        while True:
            ###################### receive input (30fps) ######################
            result = receive_images(init_variables, flag_depth)
            if result == None:
                continue

            color, depth = result

            # Get timestamp
            timestamp = NotImplemented #time.time()
            ## SI의 timestamp를 global로 접근해서 전달. 다른 idx로 변환해서 사용.

            ### Display RGBD pair ###
            cv2.imshow('RGB', color)
            if flag_depth and depth is not None:
                cv2.imshow('Depth', depth / max_depth)  # scale for visibility

            # Check for quit
            key = cv2.waitKey(1)
            if key == ord('q'):
                print("\n[Main] 'q' pressed, stopping recording...")
                break

            ###################### save asynchronously ######################
            saver.add_frame(color, depth, timestamp)

            # Print status every 30 frames
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                queue_size = saver.get_queue_size()
                print(f"[Main] RGB-D Frames: {frame_count} | FPS: {fps:.1f} | Remain Queue: {queue_size}")

    finally:
        print("\n[Main] Cleaning up...")

        # Stop audio thread first
        audio_thread.stop()

        # Stop spatial input thread
        si_thread.stop()

        # Stop saver
        saver.stop()

        sock.close()

        # Stop PV and RM Depth AHAT streams ---------------------------------------
        sink_ht, sink_pv = init_variables[0], init_variables[1]
        sink_pv.detach()
        sink_ht.detach()
        producer.stop(hl2ss.StreamPort.PERSONAL_VIDEO)
        producer.stop(hl2ss.StreamPort.RM_DEPTH_AHAT)

        # Stop PV subsystem -------------------------------------------------------
        hl2ss_lnm.stop_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)
        cv2.destroyAllWindows()

        print("[Main] Done!")


def init_hl2():
    # Start PV Subsystem ------------------------------------------------------
    hl2ss_lnm.start_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)

    # Get RM Depth AHAT calibration -------------------------------------------
    # Calibration data will be downloaded if it's not in the calibration folder
    calibration_ht = hl2ss_3dcv.get_calibration_rm(calibration_path, host, hl2ss.StreamPort.RM_DEPTH_AHAT)

    uv2xy = calibration_ht.uv2xy  # hl2ss_3dcv.compute_uv2xy(calibration_ht.intrinsics, hl2ss.Parameters_RM_DEPTH_AHAT.WIDTH, hl2ss.Parameters_RM_DEPTH_AHAT.HEIGHT)
    xy1, scale = hl2ss_3dcv.rm_depth_compute_rays(uv2xy, calibration_ht.scale)
    max_depth = calibration_ht.alias / calibration_ht.scale

    xy1_o = hl2ss_3dcv.block_to_list(xy1[:-1, :-1, :])
    xy1_d = hl2ss_3dcv.block_to_list(xy1[1:, 1:, :])

    # Start PV and RM Depth AHAT streams --------------------------------------
    producer = hl2ss_mp.producer()
    producer.configure(hl2ss.StreamPort.PERSONAL_VIDEO,
                       hl2ss_lnm.rx_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, width=pv_width, height=pv_height,
                                       framerate=pv_fps))
    producer.configure(hl2ss.StreamPort.RM_DEPTH_AHAT, hl2ss_lnm.rx_rm_depth_ahat(host, hl2ss.StreamPort.RM_DEPTH_AHAT))
    producer.initialize(hl2ss.StreamPort.PERSONAL_VIDEO, pv_fps * buffer_size)
    producer.initialize(hl2ss.StreamPort.RM_DEPTH_AHAT, hl2ss.Parameters_RM_DEPTH_AHAT.FPS * buffer_size)
    producer.start(hl2ss.StreamPort.PERSONAL_VIDEO)
    producer.start(hl2ss.StreamPort.RM_DEPTH_AHAT)

    consumer = hl2ss_mp.consumer()
    manager = mp.Manager()
    sink_pv = consumer.create_sink(producer, hl2ss.StreamPort.PERSONAL_VIDEO, manager, None)
    sink_ht = consumer.create_sink(producer, hl2ss.StreamPort.RM_DEPTH_AHAT, manager, None)

    sink_pv.get_attach_response()
    sink_ht.get_attach_response()

    # Initialize PV intrinsics and extrinsics ---------------------------------
    pv_intrinsics = hl2ss_3dcv.pv_create_intrinsics_placeholder()
    pv_extrinsics = np.eye(4, 4, dtype=np.float32)

    return [sink_ht, sink_pv, pv_intrinsics, pv_extrinsics, xy1_o, xy1_d, scale, calibration_ht], max_depth, producer


def receive_images(init_variables, flag_depth):
    sink_ht, sink_pv, pv_intrinsics, pv_extrinsics, xy1_o, xy1_d, scale, calibration_ht = init_variables

    # Get RM Depth AHAT frame and nearest (in time) PV frame --------------
    _, data_ht = sink_ht.get_most_recent_frame()
    if ((data_ht is None) or (not hl2ss.is_valid_pose(data_ht.pose))):
        return None
    _, data_pv = sink_pv.get_nearest(data_ht.timestamp)
    if ((data_pv is None) or (not hl2ss.is_valid_pose(data_pv.pose))):
        return None

    # Preprocess frames ---------------------------------------------------
    color = data_pv.payload.image

    pv_z = None
    if flag_depth:
        depth = data_ht.payload.depth  # hl2ss_3dcv.rm_depth_undistort(data_ht.payload.depth, calibration_ht.undistort_map)
        z = hl2ss_3dcv.rm_depth_normalize(depth, scale)

    # Update PV intrinsics ------------------------------------------------
    # PV intrinsics may change between frames due to autofocus
    pv_intrinsics = hl2ss_3dcv.pv_update_intrinsics(pv_intrinsics, data_pv.payload.focal_length,
                                                    data_pv.payload.principal_point)
    color_intrinsics, color_extrinsics = hl2ss_3dcv.pv_fix_calibration(pv_intrinsics, pv_extrinsics)

    # Generate depth map for PV image -------------------------------------
    if flag_depth:
        mask = (depth[:-1, :-1].reshape((-1,)) > 0)
        zv = hl2ss_3dcv.block_to_list(z[:-1, :-1, :])[mask, :]

        ht_to_pv_image = hl2ss_3dcv.camera_to_rignode(calibration_ht.extrinsics) @ hl2ss_3dcv.reference_to_world(
            data_ht.pose) @ hl2ss_3dcv.world_to_reference(data_pv.pose) @ hl2ss_3dcv.rignode_to_camera(
            color_extrinsics) @ hl2ss_3dcv.camera_to_image(color_intrinsics)

        ht_points_o = hl2ss_3dcv.rm_depth_to_points(xy1_o[mask, :], zv)
        pv_uv_o_h = hl2ss_3dcv.transform(ht_points_o, ht_to_pv_image)
        pv_list_depth = pv_uv_o_h[:, 2:]

        ht_points_d = hl2ss_3dcv.rm_depth_to_points(xy1_d[mask, :], zv)
        pv_uv_d_h = hl2ss_3dcv.transform(ht_points_d, ht_to_pv_image)
        pv_d_depth = pv_uv_d_h[:, 2:]

        mask = (pv_list_depth[:, 0] > 0) & (pv_d_depth[:, 0] > 0)

        pv_list_depth = pv_list_depth[mask, :]
        pv_d_depth = pv_d_depth[mask, :]

        pv_list_o = pv_uv_o_h[mask, 0:2] / pv_list_depth
        pv_list_d = pv_uv_d_h[mask, 0:2] / pv_d_depth

        pv_list = np.hstack((pv_list_o, pv_list_d + 1)).astype(np.int32)
        pv_z = np.zeros((pv_height, pv_width), dtype=np.float32)

        u0 = pv_list[:, 0]
        v0 = pv_list[:, 1]
        u1 = pv_list[:, 2]
        v1 = pv_list[:, 3]

        mask0 = (u0 >= 0) & (u0 < pv_width) & (v0 >= 0) & (v0 < pv_height)
        mask1 = (u1 > 0) & (u1 <= pv_width) & (v1 > 0) & (v1 <= pv_height)
        maskf = mask0 & mask1

        pv_list = pv_list[maskf, :]
        pv_list_depth = pv_list_depth[maskf, 0]

        for n in range(0, pv_list.shape[0]):
            u0 = pv_list[n, 0]
            v0 = pv_list[n, 1]
            u1 = pv_list[n, 2]
            v1 = pv_list[n, 3]

            pv_z[v0:v1, u0:u1] = pv_list_depth[n]

    return color, pv_z


def log_event(label, flag_time=False):
    global prev_label, prev

    now = time.time()
    if flag_time:
        print(f"{prev_label} ~ {label}: {now - prev:.3f}")
    prev_label = label
    prev = now


if __name__ == '__main__':
    main()
