import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# ✅ Hardcoded COCO fruit class IDs (no more coco_labels.txt issues)
custom_labels = {
    
    53: 'apple',
    55: 'orange'
}

# Load TensorFlow SSD model from TF Hub
print("Loading model...")
detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
print("Model loaded!")

def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Failed to load image: {image_path}")
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def run_detector(image_path, output_dir="outputs"):
    image_np = load_image(image_path)
    if image_np is None:
        return

    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = detector(input_tensor)

    boxes = detections['detection_boxes'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(np.int32)
    scores = detections['detection_scores'][0].numpy()

    height, width, _ = image_np.shape

    for i in range(len(scores)):
        score = scores[i]
        class_id = classes[i]

        if score > 0.5 and class_id in custom_labels:
            class_name = custom_labels[class_id]
            ymin, xmin, ymax, xmax = boxes[i]
            left, right = int(xmin * width), int(xmax * width)
            top, bottom = int(ymin * height), int(ymax * height)

            print(f"Detected: {class_name} ({score * 100:.1f}%)")

            # Draw bounding box
            cv2.rectangle(image_np, (left, top), (right, bottom), (0, 255, 0), 2)
            label = f"{class_name}: {score * 100:.1f}%"
            cv2.putText(image_np, label, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Save output
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"output_{os.path.basename(image_path)}")
    result_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, result_bgr)
    print(f"✅ Saved result to: {output_path}")

    # Show the image
    plt.figure(figsize=(12, 8))
    plt.imshow(image_np)
    plt.axis('off')
    plt.title(f"Detections for {os.path.basename(image_path)}")
    plt.show()

# Process all images in the images/ folder
image_folder = "images"
for img_file in os.listdir(image_folder):
    if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
        print(f"\n🔍 Detecting objects in {img_file}...")
        run_detector(os.path.join(image_folder, img_file))
