# GreenEye - Leaf Health Detection System

## Overview
GreenEye is an intelligent Flask web application that leverages PyTorch deep learning models to analyze leaf images. The system automatically detects whether an uploaded image contains a leaf, assesses its health status, and identifies specific plant diseases with high accuracy.
## Features
User Registration & Login: Secure authentication system

Role-based Access: Separate interfaces for users and administrators

Session Management: Secure user sessions with proper logout functionality


## Multi-Stage Analysis Pipeline
Leaf Detection - Identifies if the uploaded image contains a leaf

Health Classification - Classifies as Healthy, Dry, or Unhealthy

Disease Identification - Detects specific diseases from 40+ categories
## User Roles
 Regular Users
Upload leaf images for analysis

View prediction history

Access personal dashboard

See detailed results with confidence scores
##  Administrators
Monitor all user predictions

Delete inappropriate content

Access comprehensive analytics

Manage user-generated content

## Technical Stack

## Backend
Flask: Web framework

PyTorch: Deep learning inference

PIL/Pillow: Image processing

SQLite/PostgreSQL: Database (configurable)

## Frontend
HTML5: Structure

CSS3: Styling and responsive design

JavaScript: Dynamic interactions

AJAX: Asynchronous requests


## Machine Learning Models
MobileNetV3: Leaf detection

EfficientNet-B0: Health classification & disease detection

Custom-trained: On specialized plant disease datasets

## High Accuracy Models
Leaf Detection: 99.55% validation accuracy (MobileNetV3)

Health Classification: 99.17% validation accuracy (EfficientNet-B0)

Disease Identification: Comprehensive 40+ disease classes


##  Roadmap & Future Enhancements
Short-term Goals
Mobile application development

Additional plant species support

Real-time camera analysis

Multi-language interface

Long-term Vision
Weather integration for disease forecasting

Treatment recommendation engine

Community knowledge base

API for third-party integrations

## License
This project is licensed under the MIT License. See LICENSE file for details.

##  Authors

- [**Robiul Hasan Jisan**](https://portfolio-nine-gilt-93.vercel.app/)

  
GreenEye - Empowering plant health monitoring through artificial intelligence. Making advanced plant disease detection accessible to everyone from home gardeners to commercial farmers.

"Healthy plants, sustainable future."





