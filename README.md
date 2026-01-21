# Swift for TensorFlow Deep Learning Library

Get a taste of *protocol-oriented differentiable programming*.

This repository hosts [Swift for TensorFlow][s4tf]'s deep learning library,
available both as a part of Swift for TensorFlow toolchains and as a Swift
package.

## Usage

This library is being [automatically integrated][integrated] in Swift for
TensorFlow toolchains. You do not need to add this library as a Swift Package
Manager dependency.

### Use Google Colaboratory

[**Open an empty Colaboratory now**][blank_colab] to try out Swift,
TensorFlow, differentiable programming, and deep learning.

> For detailed usage and troubleshooting, see [Usage][usage] on the Swift for
TensorFlow project homepage.

#### Define a model

Simply import `TensorFlow` to get the full power of TensorFlow.

```swift
import TensorFlow

let hiddenSize: Int = 10

struct Model: Layer {
    var layer1 = Dense<Float>(inputSize: 4, outputSize: hiddenSize, activation: relu)
    var layer2 = Dense<Float>(inputSize: hiddenSize, outputSize: hiddenSize, activation: relu)
    var layer3 = Dense<Float>(inputSize: hiddenSize, outputSize: 3, activation: identity)
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        return input.sequenced(through: layer1, layer2, layer3)
    }
}
```

#### Initialize a model and an optimizer

```swift
var classifier = Model()
let optimizer = SGD(for: classifier, learningRate: 0.02)
Context.local.learningPhase = .training
// Dummy data.
let x: Tensor<Float> = Tensor(randomNormal: [100, 4])
let y: Tensor<Int32> = Tensor(randomUniform: [100])
```

#### Run a training loop

One way to define a training epoch is to use the
[`gradient(at:in:)`][gradient] function.

```swift
for _ in 0..<1000 {
    let 𝛁model = gradient(at: classifier) { classifier -> Tensor<Float> in
        let ŷ = classifier(x)
        let loss = softmaxCrossEntropy(logits: ŷ, labels: y)
        print("Loss: \(loss)")
        return loss
    }
    optimizer.update(&classifier, along: 𝛁model)
}
```

Another way is to make use of methods on `Differentiable` or `Layer` that
produce a backpropagation function. This allows you to compose your derivative
computation with great flexibility.

```swift
for _ in 0..<1000 {
    let (ŷ, backprop) = classifier.appliedForBackpropagation(to: x)
    let (loss, 𝛁ŷ) = valueWithGradient(at: ŷ) { ŷ in softmaxCrossEntropy(logits: ŷ, labels: y) }
    print("Model output: \(ŷ), Loss: \(loss)")
    let (𝛁model, _) = backprop(𝛁ŷ)
    optimizer.update(&classifier, along: 𝛁model)
}
```

For more models, go to [**tensorflow/swift-models**][swift-models].

## Development

Documentation covering development can be found in the [Developer Guide](Documentation/Development.md).

### OpenXLA Migration

X10 has been migrated from TensorFlow's XLA to standalone [OpenXLA](https://github.com/openxla/xla).
This provides cleaner dependencies and uses the modern PJRT runtime. See
[OpenXLA Migration Guide](docs/OPENXLA_MIGRATION.md) for details.

**Important: The Swift API is unchanged.** Your existing Swift code will work exactly as before:

```swift
import TensorFlow

// Same API - just works!
let device = Device(kind: .GPU, ordinal: 0, backend: .XLA)
let tensor = Tensor<Float>(randomNormal: [1024, 1024], on: device)
let result = matmul(tensor, tensor)
LazyTensorBarrier()  // Triggers XLA compilation via PJRT
```

The migration only affects the C++ backend - PJRT replaces XRT internally.

#### Quick Start with OpenXLA

```bash
# Prerequisites: Bazel 6.0+, C++17 compiler

# Clone and build
git clone https://github.com/tensorflow/swift-apis
cd swift-apis

# Build X10 with OpenXLA (CPU)
bazel build --config=openxla //xla_tensor:x10

# Build with GPU support
bazel build --config=openxla --config=cuda //xla_tensor:x10
```

#### Using Pre-built PJRT Plugins (Easiest)

Instead of building from source, you can use pre-built PJRT plugins from JAX or TensorFlow:

```bash
# Install JAX (provides PJRT plugins)
pip install jax[cpu]       # For CPU
pip install jax[cuda12]    # For CUDA GPU

# X10 will automatically find and use the PJRT plugin
export XLA_PLATFORM=cpu    # or "cuda" for GPU

# Find available plugins
python scripts/find_pjrt_plugin.py
```

See [Using Pre-built PJRT Plugins](Documentation/Development.md#using-pre-built-pjrt-plugins-easiest) for details.

## Bugs

Please report bugs and feature requests using GitHub issues in this repository.

## Community

Discussion about Swift for TensorFlow happens on the
[swift@tensorflow.org][forum]
mailing list.

## Contributing

We welcome contributions: please read the [Contributor Guide](CONTRIBUTING.md)
to get started. It's always a good idea to discuss your plans on the mailing
list before making any major submissions.

## Code of Conduct

In the interest of fostering an open and welcoming environment, we as
contributors and maintainers pledge to making participation in our project and
our community a harassment-free experience for everyone, regardless of age, body
size, disability, ethnicity, gender identity and expression, level of
experience, education, socio-economic status, nationality, personal appearance,
race, religion, or sexual identity and orientation.

The Swift for TensorFlow community is guided by our [Code of
Conduct](CODE_OF_CONDUCT.md), which we encourage everybody to read before
participating.

[s4tf]: https://github.com/tensorflow/swift
[integrated]: https://github.com/apple/swift/tree/tensorflow#customize-tensorflow-support
[blank_colab]: https://colab.research.google.com/notebook#create=true&language=swift
[usage]: https://github.com/tensorflow/swift/blob/main/Usage.md
[gradient]: https://www.tensorflow.org/swift/api_docs/Functions#/s:10TensorFlow8gradient2at2in13TangentVectorQzx_AA0A0Vyq_GxXEtAA14DifferentiableRzAA0aB13FloatingPointR_r0_lF
[swift-models]: https://github.com/tensorflow/swift-models
[toolchain]: https://github.com/tensorflow/swift/blob/main/Installation.md
[forum]: https://groups.google.com/a/tensorflow.org/d/forum/swift
