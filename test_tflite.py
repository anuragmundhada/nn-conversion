import numpy as np
import tensorflow as tf
import time

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="output/model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model FPS on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
start = time.time()
for idx in range(10):
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data.shape)
end = time.time()
print(end-start)


# cap = cv2.VideoCapture('/home/anurag/lightspeed/data/pushup-random.mp4')


# while(cap.isOpened()):
#     ret, frame = cap.read()

#     resized = cv2.resize(frame, (192, 192) , interpolation = cv2.INTER_LINEAR) 

#     cv2.imshow('frame', resized)


#     interpreter.set_tensor(input_details[0]['index'], resized)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
