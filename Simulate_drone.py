from cmath import inf
import sys

sys.path.append("../torch_test")

from torch_test import __version__
from torch_test.detect_kp import Angles, DetectKP, Control, Poses
import torch
from dronekit import connect, VehicleMode, mavutil
import time
import numpy as np
import cv2



torch.cuda.empty_cache()

drone_control = Control(0.8)





def wait_until(altitude, timeout, alt, period=0.25, *args, **kwargs):
  mustend = time.time() + timeout
  while time.time() < mustend:
    if altitude==alt: return True
    time.sleep(period)
  return False


def send_ned_velocity(velocity_x, velocity_y, velocity_z, duration):
    """
    Move vehicle in direction based on specified velocity vectors.
    """
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,       # time_boot_ms (not used)
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_FRAME_LOCAL_NED, # frame
        0b0000111111000111, # type_mask (only speeds enabled)
        0, 0, 0, # x, y, z positions (not used)
        velocity_x, velocity_y, velocity_z, # x, y, z velocity in m/s
        0, 0, 0, # x, y, z acceleration (not supported yet, ignored in GCS_Mavlink)
        0, 0)    # yaw, yaw_rate (not supported yet, ignored in GCS_Mavlink)


    # send command to vehicle on 1 Hz cycle
    for x in range(0,duration):
        vehicle.send_mavlink(msg)


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot open webcam")

print("Start simulator (SITL)")

connection_string = "127.0.0.1:14550"

# Connect to the Vehicle.
print("Connecting to vehicle on: %s" % (connection_string,))
vehicle = connect(connection_string, wait_ready=True)

# Get some vehicle attributes (state)
print("Get some vehicle attribute values:")
print(" GPS: %s" % vehicle.gps_0)
print(" Battery: %s" % vehicle.battery)
print(" Last Heartbeat: %s" % vehicle.last_heartbeat)
print(" Is Armable?: %s" % vehicle.is_armable)
print(" System status: %s" % vehicle.system_status.state)
print(" Mode: %s" % vehicle.mode.name)  # settable

vel_resp=np.empty((0,3))
vel_command=np.empty((0,3))
poss=np.empty((0,3))

t=np.array([])

start=time.time()
takeif=False
while True:
    
    ret, frame = cap.read()
    drone_control.detect_pose(frame)

    if drone_control.receive_commands==True and vehicle.armed==False:
        print("Arming")
        vehicle.mode = VehicleMode("GUIDED")
        vehicle.armed = True
        print("Receiving commands")

        

        while not vehicle.armed:
            print("Waiting for arming...")
            time.sleep(1)

        print("Take off")
        vehicle.simple_takeoff(10)

        

        wait_until(vehicle.location.global_relative_frame.alt,10,9.5)

    
    while drone_control.receive_commands:
        ret, frame = cap.read()
        drone_control.detect_pose(frame)

        drone_control.draw()

        x=drone_control.current_pose[Poses.X]
        y=drone_control.current_pose[Poses.Y]
        z=drone_control.current_pose[Poses.Z]

        if vehicle.mode==VehicleMode("GUIDED"):
           send_ned_velocity(x,y,z,1)

        if drone_control.receive_commands==False:
            print("Not receiving commands")
            start=time.time()
            takeif=True
            
        c = cv2.waitKey(1)
        if c == 27:
            break

    if drone_control.receive_commands==False and time.time()-start>5 and takeif==True:
        print("Landing started")
        vehicle.mode=VehicleMode("LAND")
        wait_until(vehicle.location.global_relative_frame.alt,15,0.1)
        takeif=False
        
        

    drone_control.draw()   


    c = cv2.waitKey(1)
    if c == 27:
        break


cap.release()
cv2.destroyAllWindows()

