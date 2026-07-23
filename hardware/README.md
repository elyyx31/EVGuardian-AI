# Hardware

This directory contains the hardware implementation and sensor integration details for EVGuardian AI.

## Hardware Components

- Raspberry Pi 4 Model B
- ADS1115 16-bit ADC Module
- Voltage Sensor Module
- ACS712 Current Sensor
- DS18B20 Temperature Sensor
- 3S Li-ion Battery Pack
- 3S BMS Protection Board
- Compatible Battery Charger
- DC Load
- Breadboard
- Jumper Wires

## Hardware Function

The hardware system collects real-time battery parameters such as:

- Voltage
- Current
- Temperature

The sensors continuously monitor the lithium-ion battery pack.

The Raspberry Pi acts as the main processing unit. Since some sensors provide analog signals, the ADS1115 ADC is used to convert these signals into digital values that can be read by the Raspberry Pi.

The collected battery data is processed and provided to the Machine Learning system for State of Health (SOH) estimation and Remaining Useful Life (RUL) prediction.

## Basic Hardware Flow

Battery Pack → Sensors → ADS1115 (where required) → Raspberry Pi → ML/Edge AI → Cloud → Dashboard

## Future Development

This directory will contain:

- Raspberry Pi sensor interfacing code
- Circuit connection details
- Hardware architecture
- Sensor testing programs
- Prototype images and documentation
