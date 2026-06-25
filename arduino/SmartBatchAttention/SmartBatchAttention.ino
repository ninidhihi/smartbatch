#include <Arduino.h>
#include <ArduinoBLE.h>
#include <TensorFlowLite.h>
#include <cstdarg>
#include <cstdio>

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/core/api/flatbuffer_conversions.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include "config.h"
#include "imu_scaler.h"
#include "model_data.h"

#if SMARTBATCH_IMU_REV2
#include <Arduino_BMI270_BMM150.h>
#else
#include <Arduino_LSM9DS1.h>
#endif

namespace {

constexpr float kGravity = 9.80665f;
constexpr float kDegreesToRadians = PI / 180.0f;
constexpr int kClassCount = 3;
const char* const kClassNames[kClassCount] = {
    "attentive", "drowsy", "fidgeting"};

BLEService attentionService("7d3a0001-6f5b-4e8a-9c2d-5a7b1c3e9000");
BLEUnsignedCharCharacteristic attentionScoreCharacteristic(
    "7d3a0002-6f5b-4e8a-9c2d-5a7b1c3e9000", BLERead | BLENotify);
BLEUnsignedCharCharacteristic attentionStateCharacteristic(
    "7d3a0003-6f5b-4e8a-9c2d-5a7b1c3e9000", BLERead | BLENotify);

char tflm_error[512] = {0};

class CapturingErrorReporter : public tflite::ErrorReporter {
 public:
  int Report(const char* format, va_list args) override {
    const size_t used = strlen(tflm_error);
    if (used >= sizeof(tflm_error) - 2) {
      return 0;
    }

    va_list args_copy;
    va_copy(args_copy, args);
    const int written =
        vsnprintf(tflm_error + used, sizeof(tflm_error) - used, format,
                  args_copy);
    va_end(args_copy);

    const size_t new_used = strlen(tflm_error);
    if (new_used < sizeof(tflm_error) - 1) {
      tflm_error[new_used] = '\n';
      tflm_error[new_used + 1] = '\0';
    }
    return written;
  }
};

// Harvard TinyMLx bundles TFLM 2.4, which lacks EXPAND_DIMS. For this model,
// EXPAND_DIMS is a metadata-only shape change and can use the RESHAPE kernel.
class CompatibleAllOpsResolver : public tflite::AllOpsResolver {
 public:
  CompatibleAllOpsResolver() {
    expand_dims_registration_ = tflite::ops::micro::Register_RESHAPE();
    expand_dims_registration_.builtin_code =
        tflite::BuiltinOperator_EXPAND_DIMS;
  }

  const TfLiteRegistration* FindOp(
      tflite::BuiltinOperator op) const override {
    if (op == tflite::BuiltinOperator_EXPAND_DIMS) {
      return &expand_dims_registration_;
    }
    return tflite::AllOpsResolver::FindOp(op);
  }

  const TfLiteRegistration* FindOp(const char* op) const override {
    return tflite::AllOpsResolver::FindOp(op);
  }

  BuiltinParseFunction GetOpDataParser(
      tflite::BuiltinOperator op) const override {
    if (op == tflite::BuiltinOperator_EXPAND_DIMS) {
      return tflite::ParseReshape;
    }
    return tflite::AllOpsResolver::GetOpDataParser(op);
  }

 private:
  TfLiteRegistration expand_dims_registration_;
};

alignas(16) uint8_t tensor_arena[kTensorArenaSize];
float imu_window[kWindowSamples][kImuChannels];
int samples_in_window = 0;
unsigned long next_sample_us = 0;
unsigned long last_sampling_message_ms = 0;
float latest_accel[3] = {0, 0, 0};
float latest_gyro[3] = {0, 0, 0};
bool have_accel = false;
bool have_gyro = false;

const tflite::Model* model = nullptr;
CompatibleAllOpsResolver resolver;
CapturingErrorReporter error_reporter;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
const char* model_init_error = "ERROR: Unknown model initialization failure.";

[[noreturn]] void stopWithMessage(const char* message) {
  pinMode(LED_BUILTIN, OUTPUT);
  while (true) {
    digitalWrite(LED_BUILTIN, HIGH);
    Serial.println(message);
    delay(250);
    digitalWrite(LED_BUILTIN, LOW);
    delay(1750);
  }
}

bool readImu(float sample[kImuChannels]) {
#if SMARTBATCH_IMU_REV2
  // Continuous mode fills a FIFO with synchronized accel and gyro samples.
#endif
  if (IMU.accelerationAvailable()) {
    have_accel =
        IMU.readAcceleration(latest_accel[0], latest_accel[1],
                             latest_accel[2]);
  }
  if (IMU.gyroscopeAvailable()) {
    have_gyro =
        IMU.readGyroscope(latest_gyro[0], latest_gyro[1], latest_gyro[2]);
  }

  if (!have_accel || !have_gyro) {
    return false;
  }

  sample[0] = latest_accel[0] * kGravity;
  sample[1] = latest_accel[1] * kGravity;
  sample[2] = latest_accel[2] * kGravity;
  sample[3] = latest_gyro[0] * kDegreesToRadians;
  sample[4] = latest_gyro[1] * kDegreesToRadians;
  sample[5] = latest_gyro[2] * kDegreesToRadians;
  return true;
}

int8_t quantize(float value, float scale, int zero_point) {
  int32_t quantized = static_cast<int32_t>(roundf(value / scale)) + zero_point;
  quantized = constrain(quantized, -128, 127);
  return static_cast<int8_t>(quantized);
}

float dequantize(int8_t value, float scale, int zero_point) {
  return (static_cast<int>(value) - zero_point) * scale;
}

void runInference() {
  if (input->type != kTfLiteInt8 || output->type != kTfLiteInt8) {
    Serial.println("ERROR: Model must use int8 input and output.");
    return;
  }

  if (input->bytes != kWindowSamples * kImuChannels) {
    Serial.println("ERROR: Model input is not 128 x 6 int8 values.");
    return;
  }

  for (int sample = 0; sample < kWindowSamples; ++sample) {
    for (int channel = 0; channel < kImuChannels; ++channel) {
      const float normalized =
          (imu_window[sample][channel] - imu_mean[channel]) /
          imu_std[channel];
      const int index = sample * kImuChannels + channel;
      input->data.int8[index] =
          quantize(normalized, input->params.scale, input->params.zero_point);
    }
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("ERROR: Inference failed.");
    return;
  }

  const int output_count = output->bytes;
  if (output_count != kClassCount) {
    Serial.print("ERROR: Expected 3 output classes, found ");
    Serial.println(output_count);
    return;
  }

  int best_index = 0;
  float best_score = -1000000.0f;
  float class_scores[kClassCount] = {0, 0, 0};
  float score_sum = 0.0f;
  for (int index = 0; index < output_count; ++index) {
    float score =
        dequantize(output->data.int8[index], output->params.scale,
                   output->params.zero_point);
    score = max(0.0f, score);
    class_scores[index] = score;
    score_sum += score;
    Serial.print(kClassNames[index]);
    Serial.print('=');
    Serial.print(score, 3);
    Serial.print(index + 1 == output_count ? " -> " : " ");
    if (score > best_score) {
      best_score = score;
      best_index = index;
    }
  }

  // The attention score is the normalized confidence of the attentive class.
  // 0 means no attentive confidence and 100 means fully attentive.
  const uint8_t attention_score =
      score_sum > 0.0f
          ? static_cast<uint8_t>(
                constrain(roundf(100.0f * class_scores[0] / score_sum), 0, 100))
          : 0;

  Serial.print(kClassNames[best_index]);
  Serial.print(" | attention_score=");
  Serial.println(attention_score);

  attentionScoreCharacteristic.writeValue(attention_score);
  attentionStateCharacteristic.writeValue(static_cast<uint8_t>(best_index));
}

void retainOverlap() {
  for (int sample = 0; sample < kWindowSamples - kWindowStep; ++sample) {
    for (int channel = 0; channel < kImuChannels; ++channel) {
      imu_window[sample][channel] =
          imu_window[sample + kWindowStep][channel];
    }
  }
  samples_in_window = kWindowSamples - kWindowStep;
}

bool initializeModel() {
  if (g_model_len < 1024) {
    model_init_error = "ERROR: Model data is missing or invalid.";
    return false;
  }

  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    model_init_error =
        "ERROR: Unsupported TFLite schema version. Model/library mismatch.";
    return false;
  }

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize,
      &error_reporter, nullptr);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    model_init_error = tflm_error[0] != '\0'
                           ? tflm_error
                           : "ERROR: AllocateTensors failed without details.";
    return false;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);
  Serial.print("Model arena used: ");
  Serial.print(interpreter->arena_used_bytes());
  Serial.println(" bytes");
  return true;
}

bool initializeBluetooth() {
  if (!BLE.begin()) {
    return false;
  }

  BLE.setLocalName("SmartBatch");
  BLE.setDeviceName("SmartBatch");
  BLE.setAdvertisedService(attentionService);
  attentionService.addCharacteristic(attentionScoreCharacteristic);
  attentionService.addCharacteristic(attentionStateCharacteristic);
  BLE.addService(attentionService);
  attentionScoreCharacteristic.writeValue(static_cast<uint8_t>(0));
  attentionStateCharacteristic.writeValue(static_cast<uint8_t>(0));
  BLE.advertise();

  Serial.println("Bluetooth device: SmartBatch");
  Serial.println("BLE score: 0-100; state: 0=attentive, 1=drowsy, 2=fidgeting");
  return true;
}

}  // namespace

void setup() {
  Serial.begin(115200);
  const unsigned long serial_wait_started = millis();
  while (!Serial && millis() - serial_wait_started < 5000) {
  }

  Serial.println("SmartBatch attention detector");
#if SMARTBATCH_IMU_REV2
  Serial.println("IMU: BMI270/BMM150 (Rev2)");
#else
  Serial.println("IMU: LSM9DS1 (Rev1)");
#endif

#if SMARTBATCH_IMU_REV2
  if (!IMU.begin(BOSCH_ACCELEROMETER_ONLY)) {
#else
  if (!IMU.begin()) {
#endif
    stopWithMessage("ERROR: IMU initialization failed. Check Rev1/Rev2 setting.");
  }

#if SMARTBATCH_IMU_REV2
  IMU.setContinuousMode();
  Serial.print("Accelerometer rate: ");
  Serial.print(IMU.accelerationSampleRate());
  Serial.println(" Hz");
  Serial.print("Gyroscope rate: ");
  Serial.print(IMU.gyroscopeSampleRate());
  Serial.println(" Hz");
#endif

  if (!initializeModel()) {
    stopWithMessage(model_init_error);
  }

  if (!initializeBluetooth()) {
    stopWithMessage("ERROR: Bluetooth initialization failed.");
  }

  next_sample_us = micros();
}

void loop() {
  BLE.poll();

  const unsigned long now = micros();
  if (static_cast<long>(now - next_sample_us) < 0) {
    return;
  }
  next_sample_us += kSamplePeriodUs;

  float sample[kImuChannels];
  if (!readImu(sample)) {
    if (millis() - last_sampling_message_ms >= 1000) {
      Serial.print("Waiting for IMU data: accel=");
      Serial.print(have_accel ? "OK" : "WAIT");
      Serial.print(" gyro=");
      Serial.println(have_gyro ? "OK" : "WAIT");
      last_sampling_message_ms = millis();
    }
    return;
  }

  for (int channel = 0; channel < kImuChannels; ++channel) {
    imu_window[samples_in_window][channel] = sample[channel];
  }
  ++samples_in_window;

  if (millis() - last_sampling_message_ms >= 1000) {
    Serial.print("Collecting IMU window: ");
    Serial.print(samples_in_window);
    Serial.print('/');
    Serial.println(kWindowSamples);
    last_sampling_message_ms = millis();
  }

  if (samples_in_window == kWindowSamples) {
    runInference();
    retainOverlap();
  }
}
