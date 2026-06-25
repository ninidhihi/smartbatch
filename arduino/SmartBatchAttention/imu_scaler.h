// imu_scaler.h  -- auto-generated, do not edit by hand
// Per-channel z-score parameters that MUST match training exactly.
// Apply on device BEFORE inference:  x_norm[c] = (x[c] - imu_mean[c]) / imu_std[c]
// Channel order: ax, ay, az, gx, gy, gz   (accel +-16g x/y/z, then gyro x/y/z)
// NOTE: from the SYNTHETIC-data run. Regenerate alongside model_data.h when you
// retrain on real data, or the model will see mis-scaled inputs.
#ifndef IMU_SCALER_H_
#define IMU_SCALER_H_

#define IMU_CHANNELS 6
#define IMU_WINDOW   128   // samples per window (1.28 s @ 100 Hz)

const float imu_mean[IMU_CHANNELS] = { -0.014750082f, 0.10352502f, 8.6554804f, -5.1931605e-05f, -3.366535e-05f, -8.4017611e-06f };
const float imu_std[IMU_CHANNELS]  = { 0.75802404f, 0.81962281f, 0.87776339f, 0.39625496f, 0.39621496f, 0.39620146f };

#endif  // IMU_SCALER_H_
