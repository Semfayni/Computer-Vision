import cv2
import numpy as np

net = cv2.dnn.readNetFromCaffe('data/MobileNet/mobilenet_deploy.prototxt', 'data/MobileNet/mobilenet.caffemodel')

classes = []
with open('data/MobileNet/synset.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(' ', 1)
        name = parts[1] if len(parts) > 1 else parts[0]
        classes.append(name)

filenames = ['images/MobileNet/pomegranate.jpg',
             'images/MobileNet/pen.jpg',
             'images/MobileNet/dog.jpg',
             'images/MobileNet/cat.jpeg',
             'images/MobileNet/hamster.jpg']

images = []
for filename in filenames:
    img = cv2.imread(filename)
    if img is None:
        print("Unknown")
    else:
        images.append(img)
print("")
print(f"Loading {len(images)} images")

for i, image in enumerate(images):
    filename = filenames[i]
    print("")
    print(f"Loading: {filename}")

    blob = cv2.dnn.blobFromImage(cv2.resize(image, (224, 224)),
                                 1.0 / 127.5,
                                 (224, 224),
                                 (127.5, 127.5, 127.5),
                                 swapRB=False)

    net.setInput(blob)
    preds = net.forward()
    idx = np.argmax(preds[0])
    label = classes[idx] if idx < len(classes) else "unknown"
    conf = float(preds[0][idx]) * 100
    print(f"Class: {label}")
    print(f"Probability: {round(conf, 2)} %")
    text = f"{label}: {int(conf)}%"

    cv2.putText(image, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    window_name = f"Result {i + 1}: {filename.split('/')[-1]}"
    cv2.imshow(window_name, image)

cv2.waitKey(0)
cv2.destroyAllWindows()