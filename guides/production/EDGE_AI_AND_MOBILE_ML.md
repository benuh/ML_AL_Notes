# Edge AI & Mobile ML Guide

Deploy ML models on mobile devices, IoT, and edge hardware with efficiency and speed.

## Table of Contents
1. [Why Edge AI](#why-edge-ai)
2. [Model Optimization](#model-optimization)
3. [Deployment Frameworks](#deployment-frameworks)
4. [Platform-Specific Deployment](#platform-specific-deployment)
5. [Hardware Acceleration](#hardware-acceleration)
6. [Best Practices](#best-practices)

---

## Why Edge AI

### Benefits of Edge Deployment

**Why deploy on edge:**
1. **Low Latency** - Real-time inference without network delay
2. **Privacy** - Data stays on device
3. **Offline Operation** - Works without internet
4. **Reduced Costs** - No cloud inference costs
5. **Bandwidth Savings** - No need to upload data

**Constraints:**
- **Limited compute** - CPU/GPU power
- **Memory constraints** - RAM and storage
- **Battery life** - Power efficiency critical
- **Model size** - Typically <50MB for mobile

**Edge Devices:**
- **Mobile** - Smartphones (iOS, Android)
- **IoT** - Raspberry Pi, Arduino, ESP32
- **Edge TPU** - Coral Dev Board, Google Edge TPU
- **Specialized** - NVIDIA Jetson, Intel Movidius

---

## Model Optimization

### Quantization for Edge

**Convert FP32 → INT8** for 4x size reduction and speedup.

```python
import tensorflow as tf

# Post-Training Quantization
def quantize_model_tflite(model_path, output_path):
    """Quantize TensorFlow model to INT8"""

    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)

    # Enable quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Provide representative dataset
    def representative_dataset():
        for _ in range(100):
            data = np.random.rand(1, 224, 224, 3).astype(np.float32)
            yield [data]

    converter.representative_dataset = representative_dataset

    # INT8 quantization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    # Convert
    tflite_model = converter.convert()

    # Save
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    print(f"Quantized model saved to {output_path}")

# Usage
quantize_model_tflite('saved_model/', 'quantized_model.tflite')
```

**PyTorch Quantization:**
```python
import torch
from torch.quantization import quantize_dynamic, quantize_qat

# Dynamic Quantization (easiest)
model_fp32 = YourModel()
model_int8 = quantize_dynamic(
    model_fp32,
    {torch.nn.Linear, torch.nn.LSTM},
    dtype=torch.qint8
)

# Quantization-Aware Training (best accuracy)
model_fp32.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
model_prepared = torch.quantization.prepare_qat(model_fp32.train())

# Train with quantization
for epoch in range(epochs):
    train(model_prepared)

# Convert to quantized model
model_int8 = torch.quantization.convert(model_prepared.eval())

# Export to ONNX
torch.onnx.export(
    model_int8,
    dummy_input,
    "quantized_model.onnx",
    opset_version=13
)
```

---

### Pruning for Size Reduction

```python
import torch
import torch.nn.utils.prune as prune

def prune_model(model, amount=0.3):
    """Prune model weights"""

    # Structured pruning (entire channels)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.ln_structured(
                module,
                name='weight',
                amount=amount,
                n=2,
                dim=0  # Prune output channels
            )

    # Make pruning permanent
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.remove(module, 'weight')

    return model

# Iterative magnitude pruning
def iterative_pruning(model, train_fn, target_sparsity=0.9, steps=10):
    """Gradually prune model while maintaining accuracy"""

    current_sparsity = 0
    sparsity_step = target_sparsity / steps

    for step in range(steps):
        # Increase sparsity
        current_sparsity += sparsity_step

        # Apply pruning
        prune_model(model, amount=current_sparsity)

        # Fine-tune
        train_fn(model, epochs=5)

        # Evaluate
        accuracy = evaluate(model)
        print(f"Step {step}: Sparsity {current_sparsity:.2f}, Accuracy {accuracy:.4f}")

    return model

# Usage
pruned_model = iterative_pruning(
    model,
    train_fn=lambda m, e: train(m, epochs=e),
    target_sparsity=0.9
)
```

---

### Knowledge Distillation for Edge

```python
class DistillationTrainer:
    """Train small student model from large teacher"""

    def __init__(self, teacher_model, student_model, temperature=3.0, alpha=0.7):
        self.teacher = teacher_model.eval()
        self.student = student_model
        self.temperature = temperature
        self.alpha = alpha

    def distillation_loss(self, student_logits, teacher_logits, labels):
        """Combined distillation + classification loss"""

        # Soft targets from teacher
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)

        # Distillation loss (KL divergence)
        distill_loss = F.kl_div(soft_student, soft_targets, reduction='batchmean')
        distill_loss = distill_loss * (self.temperature ** 2)

        # Hard targets (actual labels)
        student_loss = F.cross_entropy(student_logits, labels)

        # Combined loss
        return self.alpha * distill_loss + (1 - self.alpha) * student_loss

    def train(self, train_loader, epochs=10, lr=1e-3):
        optimizer = torch.optim.Adam(self.student.parameters(), lr=lr)

        for epoch in range(epochs):
            self.student.train()
            total_loss = 0

            for data, labels in train_loader:
                # Teacher predictions (no grad)
                with torch.no_grad():
                    teacher_logits = self.teacher(data)

                # Student predictions
                student_logits = self.student(data)

                # Distillation loss
                loss = self.distillation_loss(student_logits, teacher_logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}')

        return self.student

# Usage: Train MobileNet (student) from ResNet50 (teacher)
teacher = torchvision.models.resnet50(pretrained=True)
student = torchvision.models.mobilenet_v3_small(pretrained=False)

distiller = DistillationTrainer(teacher, student, temperature=3.0, alpha=0.7)
student_model = distiller.train(train_loader, epochs=20)
```

---

## Deployment Frameworks

### TensorFlow Lite

**Best for:** Android, iOS, Raspberry Pi, Microcontrollers

```python
import tensorflow as tf

# 1. Convert model to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model/')

# Optimizations
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert
tflite_model = converter.convert()

# Save
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# 2. Run inference
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prepare input
input_data = np.array(image, dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run inference
interpreter.invoke()

# Get output
output_data = interpreter.get_tensor(output_details[0]['index'])
predictions = output_data[0]
```

**Android Integration (Kotlin):**
```kotlin
// Load TFLite model
val model = LiteModel.newInstance(context)

// Prepare input
val inputFeature = TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.FLOAT32)
inputFeature.loadBuffer(byteBuffer)

// Run inference
val outputs = model.process(inputFeature)
val outputFeature = outputs.outputFeature0AsTensorBuffer

// Get predictions
val predictions = outputFeature.floatArray
```

---

### ONNX Runtime

**Best for:** Cross-platform deployment (Windows, Linux, macOS, mobile)

```python
import onnx
import onnxruntime as ort
import torch

# 1. Export PyTorch model to ONNX
model = YourModel()
dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=13,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

# 2. Optimize ONNX model
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input="model.onnx",
    model_output="model_quantized.onnx",
    weight_type=QuantType.QUInt8
)

# 3. Run inference with ONNX Runtime
session = ort.InferenceSession("model_quantized.onnx")

# Prepare input
input_name = session.get_inputs()[0].name
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)

# Run inference
outputs = session.run(None, {input_name: input_data})
predictions = outputs[0]
```

**ONNX Runtime Mobile:**
```cpp
// C++ inference on mobile
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXRuntime");
Ort::SessionOptions session_options;

// Enable optimizations
session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

// Create session
Ort::Session session(env, "model.onnx", session_options);

// Run inference
auto output_tensors = session.Run(
    Ort::RunOptions{nullptr},
    input_names, input_tensors, 1,
    output_names, 1
);
```

---

### Core ML (iOS)

**Best for:** iPhone, iPad, Mac

```python
import coremltools as ct

# 1. Convert PyTorch model to Core ML
model = YourModel().eval()
example_input = torch.rand(1, 3, 224, 224)

# Trace model
traced_model = torch.jit.trace(model, example_input)

# Convert to Core ML
mlmodel = ct.convert(
    traced_model,
    inputs=[ct.ImageType(shape=(1, 3, 224, 224), scale=1/255.0)],
    classifier_config=ct.ClassifierConfig(class_labels=['cat', 'dog'])
)

# Save
mlmodel.save("ImageClassifier.mlmodel")

# 2. Optimize for Neural Engine
mlmodel_optimized = ct.models.neural_network.quantization_utils.quantize_weights(
    mlmodel,
    nbits=8
)
mlmodel_optimized.save("ImageClassifier_quantized.mlmodel")
```

**Swift Integration:**
```swift
import CoreML
import Vision

// Load Core ML model
guard let model = try? VNCoreMLModel(for: ImageClassifier().model) else {
    fatalError("Failed to load model")
}

// Create request
let request = VNCoreMLRequest(model: model) { request, error in
    guard let results = request.results as? [VNClassificationObservation] else {
        return
    }

    // Top prediction
    if let topResult = results.first {
        print("Prediction: \(topResult.identifier), Confidence: \(topResult.confidence)")
    }
}

// Perform inference
let handler = VNImageRequestHandler(cgImage: image)
try? handler.perform([request])
```

---

## Platform-Specific Deployment

### Android ML

```kotlin
// TensorFlow Lite with GPU delegate
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate

class ImageClassifier(context: Context) {
    private val interpreter: Interpreter

    init {
        // Load model
        val model = loadModelFile(context, "model.tflite")

        // Configure interpreter
        val options = Interpreter.Options()

        // GPU acceleration
        val gpuDelegate = GpuDelegate()
        options.addDelegate(gpuDelegate)

        // NNAPI acceleration
        options.setUseNNAPI(true)

        // Multi-threading
        options.setNumThreads(4)

        interpreter = Interpreter(model, options)
    }

    fun classify(bitmap: Bitmap): String {
        // Preprocess image
        val inputBuffer = preprocessImage(bitmap)

        // Output buffer
        val outputBuffer = Array(1) { FloatArray(NUM_CLASSES) }

        // Run inference
        interpreter.run(inputBuffer, outputBuffer)

        // Post-process
        return getTopPrediction(outputBuffer[0])
    }

    private fun preprocessImage(bitmap: Bitmap): ByteBuffer {
        val inputBuffer = ByteBuffer.allocateDirect(4 * INPUT_SIZE * INPUT_SIZE * 3)
        inputBuffer.order(ByteOrder.nativeOrder())

        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)

        for (y in 0 until INPUT_SIZE) {
            for (x in 0 until INPUT_SIZE) {
                val pixel = resized Bitmap.getPixel(x, y)
                inputBuffer.putFloat((pixel shr 16 and 0xFF) / 255.0f)
                inputBuffer.putFloat((pixel shr 8 and 0xFF) / 255.0f)
                inputBuffer.putFloat((pixel and 0xFF) / 255.0f)
            }
        }

        return inputBuffer
    }
}
```

---

### iOS ML

```swift
import CoreML
import Vision

class MLInferenceService {
    private var model: VNCoreMLModel?

    init() {
        // Load Core ML model
        guard let mlModel = try? MyModel(configuration: MLModelConfiguration()) else {
            return
        }
        self.model = try? VNCoreMLModel(for: mlModel.model)
    }

    func predict(image: UIImage, completion: @escaping (String?) -> Void) {
        guard let model = model,
              let cgImage = image.cgImage else {
            completion(nil)
            return
        }

        // Create request
        let request = VNCoreMLRequest(model: model) { request, error in
            guard let results = request.results as? [VNClassificationObservation],
                  let topResult = results.first else {
                completion(nil)
                return
            }

            completion(topResult.identifier)
        }

        // Configure request
        request.imageCropAndScaleOption = .centerCrop

        // Perform inference
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        try? handler.perform([request])
    }
}

// Usage with camera
class CameraViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {
    let mlService = MLInferenceService()

    func captureOutput(_ output: AVCaptureOutput,
                      didOutput sampleBuffer: CMSampleBuffer,
                      from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }

        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let uiImage = UIImage(ciImage: ciImage)

        mlService.predict(image: uiImage) { prediction in
            print("Prediction: \(prediction ?? "unknown")")
        }
    }
}
```

---

### Raspberry Pi / Edge Devices

```python
# TensorFlow Lite on Raspberry Pi with Coral Edge TPU

from pycoral.utils import edgetpu
from pycoral.adapters import common
from pycoral.adapters import classify
from PIL import Image

# Load Edge TPU model
interpreter = edgetpu.make_interpreter('model_edgetpu.tflite')
interpreter.allocate_tensors()

# Load and preprocess image
image = Image.open('cat.jpg').resize((224, 224))
common.set_input(interpreter, image)

# Run inference
interpreter.invoke()

# Get results
classes = classify.get_classes(interpreter, top_k=5)

for c in classes:
    print(f'{labels[c.id]}: {c.score:.4f}')

# Benchmark
import time

num_iterations = 100
start = time.time()

for _ in range(num_iterations):
    interpreter.invoke()

end = time.time()
print(f'Average latency: {(end - start) / num_iterations * 1000:.2f} ms')
```

---

## Hardware Acceleration

### GPU Acceleration

```python
# TensorFlow Lite GPU Delegate
import tensorflow as tf

# Create GPU delegate
gpu_delegate = tf.lite.experimental.load_delegate('libtensorflowlite_gpu_delegate.so')

# Create interpreter with GPU
interpreter = tf.lite.Interpreter(
    model_path='model.tflite',
    experimental_delegates=[gpu_delegate]
)

interpreter.allocate_tensors()

# Run inference (on GPU)
interpreter.invoke()
```

---

### NPU/NNAPI (Android)

```kotlin
// Enable NNAPI (Neural Networks API) on Android
val options = Interpreter.Options()
options.setUseNNAPI(true)  // Use hardware accelerators

val interpreter = Interpreter(modelFile, options)
```

---

### Apple Neural Engine

```swift
// Core ML automatically uses Neural Engine when available
let config = MLModelConfiguration()
config.computeUnits = .all  // Use all available compute units (GPU + Neural Engine)

let model = try MyModel(configuration: config)
```

---

## Best Practices

### 1. Model Benchmarking

```python
class EdgeModelBenchmark:
    """Benchmark model on edge device"""

    def __init__(self, model_path, framework='tflite'):
        self.framework = framework
        self.load_model(model_path)

    def load_model(self, model_path):
        if self.framework == 'tflite':
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
        elif self.framework == 'onnx':
            self.session = ort.InferenceSession(model_path)

    def benchmark_latency(self, num_runs=100):
        """Measure inference latency"""
        import time

        latencies = []
        dummy_input = np.random.randn(1, 224, 224, 3).astype(np.float32)

        # Warmup
        for _ in range(10):
            self._inference(dummy_input)

        # Benchmark
        for _ in range(num_runs):
            start = time.time()
            self._inference(dummy_input)
            latencies.append((time.time() - start) * 1000)  # ms

        return {
            'mean_latency_ms': np.mean(latencies),
            'std_latency_ms': np.std(latencies),
            'min_latency_ms': np.min(latencies),
            'max_latency_ms': np.max(latencies),
            'p50_latency_ms': np.percentile(latencies, 50),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99)
        }

    def benchmark_memory(self):
        """Measure model size and memory usage"""
        import os
        import psutil

        model_size_mb = os.path.getsize(self.model_path) / (1024 * 1024)

        # Memory before inference
        process = psutil.Process()
        mem_before = process.memory_info().rss / (1024 * 1024)  # MB

        # Run inference
        dummy_input = np.random.randn(1, 224, 224, 3).astype(np.float32)
        self._inference(dummy_input)

        # Memory after inference
        mem_after = process.memory_info().rss / (1024 * 1024)  # MB

        return {
            'model_size_mb': model_size_mb,
            'memory_usage_mb': mem_after - mem_before
        }

    def _inference(self, input_data):
        if self.framework == 'tflite':
            self.interpreter.set_tensor(0, input_data)
            self.interpreter.invoke()
            return self.interpreter.get_tensor(self.interpreter.get_output_details()[0]['index'])
        elif self.framework == 'onnx':
            return self.session.run(None, {self.session.get_inputs()[0].name: input_data})

# Usage
benchmark = EdgeModelBenchmark('model.tflite', framework='tflite')
latency_results = benchmark.benchmark_latency(num_runs=100)
memory_results = benchmark.benchmark_memory()

print(f"Mean latency: {latency_results['mean_latency_ms']:.2f} ms")
print(f"Model size: {memory_results['model_size_mb']:.2f} MB")
```

---

### 2. On-Device Training (Federated Learning)

```python
class FederatedLearningClient:
    """On-device model updates"""

    def __init__(self, model):
        self.model = model

    def local_training(self, local_data, epochs=1):
        """Train on local device data"""
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

        for epoch in range(epochs):
            for data, labels in local_data:
                optimizer.zero_grad()
                outputs = self.model(data)
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()

        # Get updated weights
        return self.model.state_dict()

    def update_from_server(self, global_weights):
        """Update local model with global weights"""
        self.model.load_state_dict(global_weights)
```

---

## Summary

| Framework | Platform | Size | Speed | Ease of Use |
|-----------|----------|------|-------|-------------|
| **TensorFlow Lite** | Android, iOS, RPi | Small | Fast | Easy |
| **ONNX Runtime** | Cross-platform | Medium | Very Fast | Medium |
| **Core ML** | iOS, macOS | Small | Very Fast | Easy |
| **PyTorch Mobile** | Android, iOS | Medium | Fast | Medium |
| **TensorRT** | NVIDIA devices | Small | Fastest | Hard |

---

## Key Takeaways

1. **Quantization** is essential - reduces model size by 4x
2. **TensorFlow Lite** for Android/RPi, **Core ML** for iOS
3. **ONNX Runtime** for cross-platform deployment
4. **Pruning + Distillation** for maximum compression
5. **Hardware acceleration** (GPU, NPU, Neural Engine) critical
6. **Benchmark thoroughly** on target device
7. **Trade accuracy for speed** when necessary
8. **Update models** via federated learning or OTA updates

**Performance Targets:**
- **Latency:** <50ms for real-time, <200ms for interactive
- **Model Size:** <20MB for mobile, <5MB for embedded
- **Battery:** <5% battery drain per hour of active use

**Next Steps:**
- Convert your model to TFLite/ONNX/CoreML
- Quantize and benchmark on target device
- Implement hardware acceleration
- Test on real device (not simulator!)
- Optimize for production deployment
