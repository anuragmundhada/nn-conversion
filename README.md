# nn-conversion
Collection of scripts for converting trained neural networks between different libraries like pytorch, tensorflow, tensorflow-lite, core-ml, onnx etc

# Known issues
tfNCHW_to_tfNHWC adds transpose layers before operations for which tflite does not have NCHW implementations. This increases runtime significantly. Ideally, the graph needs to be re-written completely as NHWC.

