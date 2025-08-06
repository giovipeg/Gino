import serial
import threading
import time

class MCU():
    def __init__(self, port: str, baudrate: int = 500000):
        self.port = port
        self.baudrate = baudrate
        self.serial_connection = None
        
        # Store last received values
        self.last_pose_data = None  # New: roll, pitch, yaw
        self.last_accel_data = None  # Accelerometer: ax, ay, az
        self.last_gyro_data = None   # Gyroscope: gx, gy, gz
        self.last_input_data = None
        
        # Threading control
        self.running = False
        self.read_thread = None
        self.lock = threading.Lock()
        
        # Connection status
        self.arduino_ready = False
        self.imu_initialized = False

    def connect(self):
        """Establish serial connection to the MCU and start reading thread"""
        try:
            self.serial_connection = serial.Serial(self.port, self.baudrate, timeout=1)
            
            # Wait for Arduino to be ready (similar to while(!Serial) in Arduino)
            print("Waiting for Arduino to initialize...")
            time.sleep(2)  # Give Arduino time to start up
            
            # Clear any initial messages
            self.serial_connection.reset_input_buffer()
            
            self.running = True
            self.read_thread = threading.Thread(target=self._read_serial_loop, daemon=True)
            self.read_thread.start()
            
            # Wait a bit more for the thread to start processing
            time.sleep(0.5)
            
            return True
        except serial.SerialException as e:
            print(f"Failed to connect to MCU: {e}")
            return False

    def disconnect(self):
        """Stop reading thread and close serial connection"""
        self.running = False
        if self.read_thread:
            self.read_thread.join(timeout=2)
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()

    def _read_serial_loop(self):
        """Continuous thread function that reads from serial buffer"""
        while self.running:
            if not self.serial_connection or not self.serial_connection.is_open:
                time.sleep(0.1)
                continue
                
            try:
                # Check if data is available
                if self.serial_connection.in_waiting > 0:
                    line = self.serial_connection.readline().decode('utf-8').strip()
                    self._parse_line(line)
                else:
                    # Small sleep to prevent busy waiting
                    time.sleep(0.01)
                    
            except (serial.SerialException, UnicodeDecodeError) as e:
                print(f"Serial read error: {e}")
                time.sleep(0.1)

    def _parse_line(self, line):
        """Parse a single line of serial data"""
        if not line:
            return
            
        # Handle Arduino setup messages
        if "Accelerometer sample rate" in line:
            print(f"Arduino: {line}")
            self.imu_initialized = True
            return
        elif "Gyroscope sample rate" in line:
            print(f"Arduino: {line}")
            return
        elif "Failed to initialize IMU" in line:
            print(f"Arduino Error: {line}")
            return
        elif "Hz" in line and "sample rate" in line:
            print(f"Arduino: {line}")
            return
            
        # Parse POSE data (new format)
        if line.startswith('POSE:'):
            try:
                pose_data = line[5:]  # Remove 'POSE:' prefix
                values = pose_data.split(',')
                if len(values) == 3:
                    # Store as array: [roll, pitch, yaw]
                    parsed_data = [
                        float(values[0]),  # roll
                        float(values[1]),  # pitch
                        float(values[2])   # yaw
                    ]
                    with self.lock:
                        self.last_pose_data = parsed_data
                        self.arduino_ready = True
            except ValueError as e:
                print(f"Error parsing POSE data: {e}")
                
        # Parse IMU data (updated format - accelerometer and gyroscope)
        elif line.startswith('IMU;'):
            try:
                imu_data = line[4:]  # Remove 'IMU;' prefix
                values = imu_data.split(',')
                if len(values) == 6:
                    # Parse accelerometer data: [ax, ay, az]
                    accel_data = [
                        float(values[0]),  # ax
                        float(values[1]),  # ay
                        float(values[2])   # az
                    ]
                    # Parse gyroscope data: [gx, gy, gz]
                    gyro_data = [
                        float(values[3]),  # gx
                        float(values[4]),  # gy
                        float(values[5])   # gz
                    ]
                    with self.lock:
                        self.last_accel_data = accel_data
                        self.last_gyro_data = gyro_data
                        self.arduino_ready = True
            except ValueError as e:
                print(f"Error parsing IMU data: {e}")
                
        # Parse input data
        elif line.startswith('IN:'):
            try:
                input_data = line[3:]  # Remove 'IN:' prefix
                values = input_data.split(',')
                if len(values) == 3:
                    parsed_data = {
                        'button1_state': bool(int(values[0])),
                        'button2_state': bool(int(values[1])),
                        'potentiometer_value': int(values[2])
                    }
                    with self.lock:
                        self.last_input_data = parsed_data
                        self.arduino_ready = True
            except ValueError as e:
                print(f"Error parsing input data: {e}")

    def get_last_pose_data(self):
        """
        Return last known POSE data as array
        Returns: array [roll, pitch, yaw] or None if no data received yet
        """
        with self.lock:
            return self.last_pose_data.copy() if self.last_pose_data else None

    def get_last_accel_data(self):
        """
        Return last known accelerometer data as array
        Returns: array [ax, ay, az] or None if no data received yet
        """
        with self.lock:
            return self.last_accel_data.copy() if self.last_accel_data else None

    def get_last_gyro_data(self):
        """
        Return last known gyroscope data as array
        Returns: array [gx, gy, gz] or None if no data received yet
        """
        with self.lock:
            return self.last_gyro_data.copy() if self.last_gyro_data else None

    def get_last_input_data(self):
        """
        Return last known input data
        Returns: dict with keys: button1_state, button2_state, potentiometer_value or None if no data received yet
        """
        with self.lock:
            return self.last_input_data.copy() if self.last_input_data else None

    def is_connected(self):
        """Check if MCU is connected and reading thread is running"""
        return (self.serial_connection and 
                self.serial_connection.is_open and 
                self.running and 
                self.read_thread and 
                self.read_thread.is_alive())

    def is_arduino_ready(self):
        """Check if Arduino has sent its first data (indicating it's ready)"""
        with self.lock:
            return self.arduino_ready

    def is_imu_initialized(self):
        """Check if IMU was successfully initialized on Arduino"""
        return self.imu_initialized


# Simple usage example
if __name__ == "__main__":
    # Initialize MCU (adjust port as needed)
    mcu_device = MCU('/dev/ttyACM0', 500000)
    
    # Connect to MCU
    if mcu_device.connect():
        print("MCU connected successfully!")
        
        # Wait for Arduino to be ready
        print("Waiting for Arduino to send first data...")
        while not mcu_device.is_arduino_ready():
            time.sleep(0.1)
        print("Arduino is ready!")
        
        try:
            # Main loop - continuously read data
            while True:
                # Get latest POSE data
                pose_data = mcu_device.get_last_pose_data()
                if pose_data:
                    print(f"POSE: roll={pose_data[0]:.2f}, pitch={pose_data[1]:.2f}, yaw={pose_data[2]:.2f}")
                
                # Get latest accelerometer data
                accel_data = mcu_device.get_last_accel_data()
                if accel_data:
                    print(f"Accel: ax={accel_data[0]:.2f}, ay={accel_data[1]:.2f}, az={accel_data[2]:.2f}")
                
                # Get latest gyroscope data
                gyro_data = mcu_device.get_last_gyro_data()
                if gyro_data:
                    print(f"Gyro: gx={gyro_data[0]:.2f}, gy={gyro_data[1]:.2f}, gz={gyro_data[2]:.2f}")
                
                # Get latest input data
                input_data = mcu_device.get_last_input_data()
                if input_data:
                    print(f"Inputs: Button1={input_data['button1_state']}, Button2={input_data['button2_state']}, Pot={input_data['potentiometer_value']}")
                
                # Check connection status
                if not mcu_device.is_connected():
                    print("MCU connection lost!")
                    break
                
                time.sleep(0.01)  # Wait 100ms between readings
                
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            # Clean shutdown
            mcu_device.disconnect()
            print("MCU disconnected.")
    else:
        print("Failed to connect to MCU!")
