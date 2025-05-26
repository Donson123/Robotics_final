import cv2
import numpy as np
from picamera2 import Picamera2
import time
import matplotlib.pyplot as plt
import math
from motor import tankMotor  # Controls the robot's motors
from Grab import Servo

# Capture N images over T seconds and average them to reduce noise
def capture_average_image(camera, filename, num_images=5, total_duration=10):
    images = []
    interval = total_duration / num_images

    print(f"Capturing {num_images} background images over {total_duration} seconds...")
    for i in range(num_images):
        time.sleep(interval)
        img = camera.capture_array()
        img = cv2.flip(img, -1)
        images.append(img.astype(np.float32))
        print(f"Captured background image {i+1}/{num_images}")

    avg_image = np.mean(images, axis=0).astype(np.uint8)
    cv2.imwrite(filename, avg_image)
    return avg_image

# Capture a single scene image
def capture_image(camera, filename):
    time.sleep(2)
    image = camera.capture_array()
    image = cv2.flip(image, -1)
    cv2.imwrite(filename, image)
    return image

# Detect new objects based on background subtraction
def detect_objects(background_img, scene_img):
    gray_bg = cv2.cvtColor(background_img, cv2.COLOR_BGR2GRAY)
    gray_scene = cv2.cvtColor(scene_img, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gray_scene, gray_bg)
    _, thresh = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)

    thresh = cv2.GaussianBlur(thresh, (5, 5), 0)
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort coutours and get the biggest one
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    object_centers = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w // 2, y + h // 2
            object_centers.append((cx, cy))
            # draw bounding box
            cv2.rectangle(scene_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(scene_img, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(scene_img, f"{cx},{cy}", (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            break  # only keep the biggest one

    return scene_img, object_centers


# Convert pixel to meter coordinates 
def pixel_to_meter(px, py, origin_y=480, ppm=1000):
    x_m = (px - 320) / ppm
    y_m = (origin_y - py) / ppm
    return round(x_m, 2), round(y_m, 2)

# Draw top-down view and save to file
def generate_topdown_plot(object_centers_px, filename="challenge1_view.png"):
    ppm = 1000
    object_positions_m = []

    for (px, py) in object_centers_px:
        x_m, y_m = pixel_to_meter(px, py, ppm=ppm)
        object_positions_m.append((x_m, y_m))

    plt.figure(figsize=(6, 4))
    plt.title("2D Top-Down View")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.grid(True)

    plt.plot(0, 0, 'bo', label="Camera (0, 0)")

    for i, (x, y) in enumerate(object_positions_m):
        plt.plot(x, y, 'rs')
        plt.text(x + 0.05, y, f"Obj {i+1}", fontsize=10)

    plt.plot([0.5, 1.0], [-0.2, -0.2], 'k-', linewidth=2)
    plt.text(0.6, -0.25, "0.5 m", fontsize=9)

    plt.axis('equal')
    plt.xlim(-1, 1.5)
    plt.ylim(-0.5, 1.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Let the robot rotate and drive to a target point
def drive_to_target(x_m, y_m):
    motor = tankMotor()

    angle_deg = math.degrees(math.atan2(x_m, y_m))
    print(f"Rotating to {angle_deg:.1f}°")

    if abs(angle_deg) > 5:
        if angle_deg > 0:
            motor.setMotorModel(-1000, 1000)  # Left
        else:
            motor.setMotorModel(1000, -1000)  # Right
        time.sleep(min(abs(angle_deg) / 90, 1.5))

    motor.setMotorModel(0, 0)
    time.sleep(0.2)

    print(f"Driving forward {y_m:.2f} m")
    motor.setMotorModel(1200, 1200)  # Forward

    drive_speed = 0.40  # 40 cm in 1 second with PWM 1200
    duration = y_m / drive_speed
    time.sleep(min(duration, 4.0))  # Max 4 sec

    motor.setMotorModel(0, 0)
    motor.close()

def main():
    cam = Picamera2()
    cam.preview_configuration.main.size = (640, 480)
    cam.preview_configuration.main.format = "RGB888"
    cam.configure("preview")
    cam.start()

    # get background
    background = capture_average_image(cam, "background.png", num_images=5, total_duration=8)

    input("press enter")

    # get image with new objects
    scene = capture_image(cam, "scene_with_objects.png")

    # search objects
    result_img, centers = detect_objects(background, scene)
    print(" objectcenter(s) (in pixels):", centers)

    if len(centers) == 0:
        print("No object detected")
    else:

        servo = Servo()
        servo.toOrigin() 
        # get the object
        target_px = centers[0]
        x_m, y_m = pixel_to_meter(*target_px)
        print(f"position in: x = {x_m:.2f}, y = {y_m:.2f}")

        # Afstand vóór het object om te stoppen
        stop_margin = -0.1  # meters

        # Corrigeer y zodat robot net vóór het object stopt
        target_y_m = max(0.0, y_m - stop_margin)


        generate_topdown_plot([target_px], "topdown_view.png")
        print("Top-down view saved as topdown_view.png")


        initial_angle = math.degrees(math.atan2(x_m, target_y_m))
        initial_distance = target_y_m

        drive_to_target(x_m, target_y_m)
        servo.grab()

        print("back to startposition")

        motor = tankMotor()

        # same distance back
        drive_speed = 0.40
        duration = initial_distance / drive_speed
        motor.setMotorModel(-1200, -1200)  # backwards
        time.sleep(min(duration, 4.0))

        motor.setMotorModel(0, 0)
        motor.close()

        if abs(initial_angle) > 5:
            print(f"original angle: {-initial_angle:.1f}°")
            motor = tankMotor()
            if initial_angle > 0:
                motor.setMotorModel(1000, -1000)  # right
            else:
                motor.setMotorModel(-1000, 1000)  #  left
            time.sleep(abs(initial_angle) / 90)  # 90° ~ 1 sec
            motor.setMotorModel(0, 0)
            motor.close()

        # turn 180 degrees and drop item
        motor = tankMotor()
        motor.setMotorModel(-1000, 1000) 
        time.sleep(1.75)  # takes around this time
        motor.setMotorModel(0, 0)
        time.sleep(0.5)

        # drop
        servo.toOrigin()  # back to original position
        time.sleep(0.5)

        # turn back 180 degrees
        motor.setMotorModel(1000, -1000) 
        time.sleep(1.75)
        motor.setMotorModel(0, 0)
        motor.close()

    # save result
    cv2.imwrite("result.png", result_img)

if __name__ == "__main__":
    main()

