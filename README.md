# Lip Reading AI Application

This project is a web application that uses deep learning to perform lip reading on video inputs. It consists of a React frontend and a FastAPI backend, both containerized with Docker.

## Project Structure

- `backend/`: FastAPI backend that processes videos and makes predictions using the lip reading model
- `whisper-words-flow/`: React frontend for uploading videos and displaying predictions
- `Lip_Reading_Using_Deep_Learning/`: Contains the pre-trained lip reading model

## Prerequisites

- Docker and Docker Compose
- The pre-trained model weights in `Lip_Reading_Using_Deep_Learning/models - checkpoint 96/`

## Getting Started

1. Clone this repository
2. Make sure you have the pre-trained model weights in the correct location
3. Run the application using Docker Compose:

```bash
docker-compose up -d
```

4. Access the application at http://localhost

### Using the Dummy Model

If you don't have the pre-trained model weights, you can still run the application with a dummy model that returns fixed predictions. This is enabled by default in the docker-compose.yml file with the `USE_DUMMY_MODEL=true` environment variable.

To use the real model, set `USE_DUMMY_MODEL=false` in the docker-compose.yml file.

## How It Works

1. Upload a video file containing lip movements
2. The frontend sends the video to the backend API
3. The backend processes the video frames, focusing on the lip region
4. The pre-trained deep learning model analyzes the lip movements and predicts the spoken words
5. The prediction is sent back to the frontend and displayed to the user

## Model Architecture

The lip reading model uses a 3D CNN (Convolutional Neural Network) combined with Bidirectional LSTM (Long Short-Term Memory) layers to analyze the temporal patterns in lip movements. The model was trained on a dataset of aligned video and text pairs.

## Development

### Backend

The backend is built with FastAPI and uses TensorFlow for the deep learning model. To run the backend locally:

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend

The frontend is built with React, TypeScript, and Tailwind CSS. To run the frontend locally:

```bash
cd whisper-words-flow
npm install
npm run dev
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
