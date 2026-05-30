#include <Arduino.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include <AccelStepper.h>

// =====================================================
// BLE UUIDs
// =====================================================

#define NUS_SERVICE_UUID "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
#define NUS_TX_CHAR_UUID "6e400003-b5a3-f393-e0a9-e50e24dcca9e"
#define NUS_RX_CHAR_UUID "6e400002-b5a3-f393-e0a9-e50e24dcca9e"

// =====================================================
// EMG SENSOR
// =====================================================

#define EMG_PIN 36

// =====================================================
// A4988 + NEMA17
// =====================================================

#define STEP_PIN 26
#define DIR_PIN  27

// =====================================================
// MOTOR
// =====================================================

AccelStepper stepper(
  AccelStepper::DRIVER,
  STEP_PIN,
  DIR_PIN
);

// =====================================================
// BLE GLOBALS
// =====================================================

BLEServer* gServer = nullptr;

BLECharacteristic* gTxChar = nullptr;
BLECharacteristic* gRxChar = nullptr;

bool gConnected = false;

// =====================================================
// EMG VARIABLES
// =====================================================

float baseline = 0;

float envelope = 0;

const float ALPHA = 0.08;

// =====================================================
// MOTOR VARIABLES
// =====================================================

float currentAngle = 0;

const float STEPS_PER_REV = 3200.0;

const float STEPS_PER_DEGREE =
    STEPS_PER_REV / 360.0;

// =====================================================
// BLE CALLBACKS
// =====================================================

class ServerCallbacks : public BLEServerCallbacks {

  void onConnect(BLEServer*) override {

    gConnected = true;

    Serial.println("[BLE] Connected");
  }

  void onDisconnect(BLEServer* server) override {

    gConnected = false;

    Serial.println("[BLE] Disconnected");

    server->getAdvertising()->start();
  }
};

// =====================================================
// RX CALLBACKS
// =====================================================

class RXCallbacks :
    public BLECharacteristicCallbacks {

  void onWrite(
      BLECharacteristic* characteristic
  ) override {

    String value =
        characteristic->getValue().c_str();

    value.trim();

    if (value.length() == 0)
      return;

    Serial.print("[RX] ");
    Serial.println(value);

    // =============================================
    // TARGET ANGLE
    // =============================================

    float targetAngle =
        value.toFloat();

    targetAngle =
        constrain(
            targetAngle,
            0,
            130
        );

    long targetSteps =
        targetAngle *
        STEPS_PER_DEGREE;

    stepper.moveTo(targetSteps);

    currentAngle = targetAngle;
  }
};

// =====================================================
// BLE SETUP
// =====================================================

void setupBLE() {

  BLEDevice::init("NeuroFlex-EMG");

  BLEDevice::setMTU(247);

  gServer =
      BLEDevice::createServer();

  gServer->setCallbacks(
      new ServerCallbacks()
  );

  BLEService* service =
      gServer->createService(
          NUS_SERVICE_UUID
      );

  // =============================================
  // TX
  // =============================================

  gTxChar =
      service->createCharacteristic(
          NUS_TX_CHAR_UUID,
          BLECharacteristic::PROPERTY_NOTIFY
      );

  gTxChar->addDescriptor(
      new BLE2902()
  );

  // =============================================
  // RX
  // =============================================

  gRxChar =
      service->createCharacteristic(
          NUS_RX_CHAR_UUID,
          BLECharacteristic::PROPERTY_WRITE |
          BLECharacteristic::PROPERTY_WRITE_NR
      );

  gRxChar->setCallbacks(
      new RXCallbacks()
  );

  service->start();

  BLEAdvertising* adv =
      BLEDevice::getAdvertising();

  adv->addServiceUUID(
      NUS_SERVICE_UUID
  );

  adv->setScanResponse(true);

  BLEDevice::startAdvertising();

  Serial.println("[BLE] Advertising...");
}

// =====================================================
// CALIBRATION
// =====================================================

void calibrateEMG() {

  Serial.println(
      "\n[CALIBRATION]"
  );

  Serial.println(
      "Relax your muscle..."
  );

  delay(3000);

  long total = 0;

  const int samples = 1000;

  for (int i = 0; i < samples; i++) {

    total += analogRead(EMG_PIN);

    delay(2);
  }

  baseline =
      total / (float)samples;

  Serial.print("Baseline = ");

  Serial.println(baseline);

  Serial.println(
      "[CALIBRATION DONE]\n"
  );
}

// =====================================================
// READ EMG
// =====================================================

float readEMG() {

  int raw =
      analogRead(EMG_PIN);

  // =============================================
  // REMOVE DC OFFSET
  // =============================================

  float signal =
      raw - baseline;

  // =============================================
  // RECTIFY
  // =============================================

  signal = abs(signal);

  // =============================================
  // LOW PASS ENVELOPE
  // =============================================

  envelope =
      (ALPHA * signal) +
      ((1.0 - ALPHA) * envelope);

  return envelope;
}

// =====================================================
// NORMALIZE
// =====================================================

int normalizeEMG(float value) {

  int pct =
      map(
          (int)value,
          0,
          1200,
          0,
          100
      );

  pct =
      constrain(
          pct,
          0,
          100
      );

  return pct;
}

// =====================================================
// SENSOR STATUS
// =====================================================

String getStatus(float value) {

  if (value < 5)
    return "DISCONNECTED";

  if (value < 15)
    return "WEAK";

  if (value > 2500)
    return "NOISY";

  return "OK";
}

// =====================================================
// SETUP
// =====================================================

void setup() {

  Serial.begin(115200);

  delay(1000);

  // =============================================
  // ADC
  // =============================================

  analogReadResolution(12);

  analogSetAttenuation(
      ADC_11db
  );

  pinMode(EMG_PIN, INPUT);

  // =============================================
  // MOTOR
  // =============================================

  stepper.setMaxSpeed(2500);

  stepper.setAcceleration(1500);

  // =============================================
  // CALIBRATION
  // =============================================

  calibrateEMG();

  // =============================================
  // BLE
  // =============================================

  setupBLE();

  Serial.println(
      "[SYSTEM READY]"
  );
}

// =====================================================
// LOOP
// =====================================================

void loop() {

  // =============================================
  // MOTOR
  // =============================================

  stepper.run();

  // =============================================
  // SAMPLE RATE
  // =============================================

  static uint32_t lastSample = 0;

  if (micros() - lastSample < 1000)
    return;

  lastSample = micros();

  // =============================================
  // REAL EMG SIGNAL
  // =============================================

  float emg =
      readEMG();

  // =============================================
  // SIMULATED MUSCLE CHANNELS
  // =============================================

  float bf =
      emg * 0.85;

  float rf =
      emg * 1.10;

  float st =
      emg * 0.70;

  float vm =
      emg * 1.25;

  // =============================================
  // PERCENTAGES
  // =============================================

  int bfPct =
      normalizeEMG(bf);

  int rfPct =
      normalizeEMG(rf);

  int stPct =
      normalizeEMG(st);

  int vmPct =
      normalizeEMG(vm);

  // =============================================
  // STATUS
  // =============================================

  String bfStatus =
      getStatus(bf);

  String rfStatus =
      getStatus(rf);

  String stStatus =
      getStatus(st);

  String vmStatus =
      getStatus(vm);

  // =============================================
  // JSON
  // =============================================

  char json[512];

  snprintf(
      json,
      sizeof(json),

      "{"

      "\"raw\":%.2f,"

      "\"bicepsfemoris\":%.2f,"
      "\"rectusfemoris\":%.2f,"
      "\"semitendinosus\":%.2f,"
      "\"vastusmedialis\":%.2f,"

      "\"bf_pct\":%d,"
      "\"rf_pct\":%d,"
      "\"st_pct\":%d,"
      "\"vm_pct\":%d,"

      "\"bf_status\":\"%s\","
      "\"rf_status\":\"%s\","
      "\"st_status\":\"%s\","
      "\"vm_status\":\"%s\","

      "\"motor_angle\":%.2f"

      "}",

      emg,

      bf,
      rf,
      st,
      vm,

      bfPct,
      rfPct,
      stPct,
      vmPct,

      bfStatus.c_str(),
      rfStatus.c_str(),
      stStatus.c_str(),
      vmStatus.c_str(),

      currentAngle
  );

  // =============================================
  // BLE SEND
  // =============================================

  if (gConnected && gTxChar) {

    gTxChar->setValue(
        (uint8_t*)json,
        strlen(json)
    );

    gTxChar->notify();
  }

  // =============================================
  // SERIAL DEBUG
  // =============================================

  static uint32_t lastPrint = 0;

  if (millis() - lastPrint > 50) {

    Serial.println(json);

    lastPrint = millis();
  }
}