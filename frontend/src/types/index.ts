
export interface PredictionResult {
  id: string;
  timestamp: Date;
  videoName: string;
  prediction: string;
  confidence?: number;
  videoUrl: string;
}
