# Arduino IDE guide

## 1. Install Arduino IDE and the board

1. Install Arduino IDE 2.
2. Open **Tools > Board > Boards Manager**.
3. Search for `Arduino Mbed OS Nano Boards`.
4. Install it.
5. Connect the Nano 33 BLE Sense by USB.
6. Select **Tools > Board > Arduino Mbed OS Nano Boards > Arduino Nano 33 BLE**.
7. Select the board's serial port under **Tools > Port**.

The board menu can say Nano 33 BLE even when the physical board is the Sense
variant.

## 2. Identify the IMU revision

- Nano 33 BLE Sense Rev1 uses the LSM9DS1.
- Nano 33 BLE Sense Rev2 uses the BMI270 plus BMM150.

The underside of a Rev2 board is normally marked `REV2`. Do not combine
calibration data from Rev1 and Rev2 boards.

Open `arduino/SmartBatchAttention/config.h` and set exactly one revision:

```cpp
#define SMARTBATCH_IMU_REV2 1
```

Use `1` for Rev2 and `0` for Rev1.

## 3. Install libraries

Open **Tools > Manage Libraries** and install:

- Rev2: `Arduino_BMI270_BMM150`
- Rev1: `Arduino_LSM9DS1`

TensorFlow's official Arduino repository currently documents manual
installation. Close Arduino IDE, open PowerShell, and run:

```powershell
Set-Location "$HOME\Documents\Arduino\libraries"
git clone https://github.com/tensorflow/tflite-micro-arduino-examples Arduino_TensorFlowLite
```

If your Arduino sketchbook is in another directory, use its `libraries`
subdirectory instead. Restart Arduino IDE afterward. You should see
`Arduino_TensorFlowLite` under **File > Examples**.

## 4. Generate model and scaler headers

On the computer, open PowerShell in this repository and run:

```powershell
python tools\make_arduino_artifacts.py `
  --model C:\path\to\attention_model_int8.tflite `
  --scaler C:\path\to\imu_scaler.npz `
  --output arduino\SmartBatchAttention
```

This replaces the placeholder `model_data.h` and `imu_scaler.h`. The tool also
checks that the model input is int8 and has 768 elements, which corresponds to
`128 x 6`.

## 5. Open and upload

1. In Arduino IDE, choose **File > Open**.
2. Open `arduino/SmartBatchAttention/SmartBatchAttention.ino`.
3. Click **Verify**.
4. Click **Upload**.
5. Open **Tools > Serial Monitor** and select `115200` baud.
6. Hold the board steadily against the chest.

After 1.28 seconds the sketch prints scores and a predicted class every
0.64 seconds. The labels are currently:

```text
attentive, drowsy, fidgeting
```

Change `kClassNames` in the sketch if the trained model uses another label
order. The order must exactly match the Python training labels.

## 6. Save a run for Excel

Close Serial Monitor before starting the logger, because only one program can
usually read the Arduino serial port at a time.

From this repository, run:

```powershell
python tools\log_attention_output.py --port COM3 --duration 300
```

Replace `COM3` with the port selected in Arduino IDE. `--duration 300` records
for five minutes. Omit it to record until you press `Ctrl+C`.

The logger creates three files under `outputs\attention_runs\`:

- `.txt`: the complete raw Arduino output
- `.csv`: rows that Excel can open directly
- `.xlsx`: an Excel workbook with the parsed prediction rows

Each spreadsheet row contains:

```text
timestamp, elapsed_seconds, attentive_score, drowsy_score, fidgeting_score,
prediction, attention_score, raw_line
```

## 7. Expected serial output

```text
SmartBatch attention detector
IMU: BMI270/BMM150 (Rev2)
Model arena used: 73120 bytes
attentive=0.812 drowsy=0.133 fidgeting=0.055 -> attentive
```

The exact arena value and scores depend on the model.

## Important accuracy notes

- The Python scaler values must be copied without alteration.
- Arduino acceleration is converted from `g` to `m/s^2`.
- Arduino angular velocity is converted from degrees per second to `rad/s`.
- Mount the badge in the same orientation used during badge-data collection.
- Rev1 and Rev2 require separate sensor calibration and should not share
  recordings blindly.
- This sketch implements the IMU attention model only. Microphone log-mel
  extraction and multimodal fusion are a separate RAM-sensitive stage.

## Common errors

`MODEL HEADER IS A PLACEHOLDER`

: Run `tools/make_arduino_artifacts.py` with the real model and scaler.

`AllocateTensors failed`

: Increase `kTensorArenaSize` cautiously. The nRF52840 has 256 KB RAM, and the
  arena must leave room for the stack, sensor libraries, and buffers.

`Model input must contain 128 x 6 values`

: The selected `.tflite` file does not match this IMU deployment contract.

No serial port after upload

: Double-press reset to enter the bootloader, select the newly appearing port,
  and upload again.
