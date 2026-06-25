#pragma once

// Set to 1 for Nano 33 BLE Sense Rev2, or 0 for Rev1.
#define SMARTBATCH_IMU_REV2 0
#define SMARTBATCH_BLE_NAME "SmartBatch"
// The handoff contract uses 100 Hz and a 128-sample window.
constexpr unsigned long kSamplePeriodUs = 10000;
constexpr int kWindowSamples = 128;
constexpr int kImuChannels = 6;
constexpr int kWindowStep = 64;

// Increase only if AllocateTensors() fails.
constexpr int kTensorArenaSize = 120 * 1024;
