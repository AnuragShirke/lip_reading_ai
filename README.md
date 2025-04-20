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

This repository is set up for deployment on Render using a Blueprint.

### Option 1: Deploy with Render Blueprint (Recommended)

1. Fork this repository to your GitHub account
2. Sign up for [Render](https://render.com) if you haven't already
3. Click the "New +" button and select "Blueprint"
4. Connect your GitHub account and select this repository
5. Render will automatically deploy both the frontend and backend services

### Option 2: Manual Deployment

#### Backend Deployment

1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Configure the service:
   - Name: lip-reading-backend
   - Environment: Python
   - Build Command: `pip install -r backend/requirements.txt`
   - Start Command: `cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT`
   - Environment Variables:
     - `USE_DUMMY_MODEL`: false
     - `HF_REPO_ID`: AnuragShirke/lip-reading-models

#### Frontend Deployment

1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Configure the service:
   - Name: lip-reading-frontend
   - Environment: Node
   - Build Command: `cd frontend && npm install && npm run build`
   - Start Command: `cd frontend && npm start`
   - Environment Variables:
     - `BACKEND_URL`: https://lip-reading-backend.onrender.com (replace with your backend URL)

## Local Development

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

## License

MIT
