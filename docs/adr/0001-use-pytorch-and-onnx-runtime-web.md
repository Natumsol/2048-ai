# Use PyTorch and ONNX Runtime Web for AI inference

Train the 2048 policy model with PyTorch, export it as ONNX, and run it locally in the browser with ONNX Runtime Web. This replaces the original Keras and Keras.js constraint because Keras.js has an uncertain compatibility path with current Keras model formats, while ONNX provides a maintained boundary between offline training and the PixiJS application.
