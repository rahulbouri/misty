# Misty II Robotics AI Toolkit

This repository is dedicated to enhancing the capabilities of the Misty II robot through various implementations and tools. We have created deep learning pipeline for various tasks. The project was taken up with the goal of creating a Robot capable of Navigating Indoor environments, Interact with human like capabilities and perform Sentiment analysis.

## Table of Contents 

- [Installation](#installation)
- [Interaction Pipeline](#interactionpipeline)
- [Fall and Pose Detection](#fallposeanddetection)
- [Vision based Navigation System](#visionbasednavigationsystem)

## Installation

This repository contains tools and implementations for programming the Misty II robot from Misty Robotics. The base [repository](https://github.com/MistyCommunity/Python-SDK) includes the Python SDK toolkit for Misty. Please refer to the repository mentioned above for proper installation of base environment requirements.

Please contact the owner of this repository for further details

## Interaction Pipeline

The interaction pipeline utilizes a custom-made speech recognition model to decipher user queries. These queries are then passed through a fine-tuned language model (LLM) trained on proprietary datasets. The response is generated using the SpeechT5 text-to-speech (TTS) model, ensuring a natural-sounding reply.

## Fall and Pose Detection

Using Google MediaPipe, a pipeline has been developed for fall detection. Upon detecting a fall, the system moves towards the fallen person and attempts to establish a proper response. If necessary, emergency services are contacted.

## Vision based Navigation System

Utilizing bounding boxes from the YOLOv4 model and point cloud generation (calibration code for Misty's camera is provided in the repository), this implementation enables vision-based navigation. By calculating appropriate traversing distances, Misty II can navigate its environment effectively.

Note:
This repository is a work in progress and may undergo further updates and enhancements. For more information or assistance, please reach out to the repository owner.
