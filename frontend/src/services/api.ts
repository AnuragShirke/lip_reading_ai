// API service for communicating with the backend

interface PredictionResponse {
  prediction: string;
  confidence: number;
  video_url?: string; // URL to the processed video
  thumbnail_url?: string; // URL to the video thumbnail
}

// Always use localhost:8000 when accessing from a browser
const API_URL = 'http://localhost:8000';

console.log('Using API URL:', API_URL);

export const uploadVideo = async (file: File): Promise<PredictionResponse> => {
  try {
    console.log('Uploading video to:', `${API_URL}/predict`);
    console.log('File type:', file.type);
    console.log('File size:', file.size, 'bytes');

    const formData = new FormData();
    formData.append('file', file);

    console.log('Sending fetch request...');
    const response = await fetch(`${API_URL}/predict`, {
      method: 'POST',
      body: formData,
      // Add CORS headers
      mode: 'cors',
      credentials: 'same-origin',
      headers: {
        'Accept': 'application/json',
      },
    }).catch(error => {
      console.error('Fetch error:', error);
      throw new Error(`Network error: ${error.message}`);
    });

    console.log('Response received:', response.status, response.statusText);

    if (!response.ok) {
      let errorMessage = 'Failed to process video';
      try {
        const errorData = await response.json();
        console.error('Error data:', errorData);
        errorMessage = errorData.detail || errorMessage;
      } catch (e) {
        // If we can't parse the JSON, just use the status text
        console.error('Error parsing error response:', e);
        errorMessage = response.statusText || errorMessage;
      }
      throw new Error(errorMessage);
    }

    return await response.json();
  } catch (error) {
    console.error('Error uploading video:', error);
    throw error;
  }
};
