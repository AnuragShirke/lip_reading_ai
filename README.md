# Lip Reading AI Application

This is a lip reading application that uses deep learning to predict spoken words from video clips of lip movements.

## Features

- Upload videos of lip movements
- Get real-time text predictions
- Support for various video formats (MP4, MPG, WebM, MOV, AVI)
- Dark/Light theme toggle
- History of previous predictions

## Technologies Used

- **Frontend**: React, TypeScript, Tailwind CSS
- **Backend**: Python, FastAPI
- **ML Model**: TensorFlow/Keras
- **Containerization**: Docker

## Deployment

This repository is set up for deployment on free hosting platforms like Render.

### Frontend

The frontend is a React application built with Vite and TypeScript.

### Backend

The backend is a FastAPI application that processes videos and runs the lip reading model.

## Getting Started

1. Clone this repository
2. Run `docker-compose up -d` to start the application
3. Open your browser and navigate to `http://localhost`

## License

MIT
