# Smart Classroom Attention Detector

This workspace contains the Arduino deployment side of the chest-mounted
attention detector. It preserves the training contract from the handoff:

- 128 IMU samples per window
- 6 channels ordered `ax, ay, az, gx, gy, gz`
- accelerometer values in `m/s^2`
- gyroscope values in `rad/s`
- train-set scaler copied from `imu_scaler.npz`
- full int8 TensorFlow Lite Micro inference

The sketch is in `arduino/SmartBatchAttention/`.

## What is still required

The trained files were not present in this workspace. Before the sketch can run
inference, obtain:

1. `attention_model_int8.tflite`
2. `imu_scaler.npz`

Then generate the Arduino headers:

```powershell
python tools\make_arduino_artifacts.py `
  --model path\to\attention_model_int8.tflite `
  --scaler path\to\imu_scaler.npz `
  --output arduino\SmartBatchAttention
```

See [ARDUINO_IDE_GUIDE.md](ARDUINO_IDE_GUIDE.md) for installation, board
revision selection, upload, and troubleshooting.

## Saving prediction results

After uploading the sketch, keep the board connected and capture the Serial
Monitor output from the computer:

```powershell
python tools\log_attention_output.py --port COM3 --duration 300
```

Replace `COM3` with the Arduino port shown in Arduino IDE. The command saves:

- a raw `.txt` serial log
- a structured `.csv`
- an Excel `.xlsx` workbook

The files are written under `outputs\attention_runs\`.
