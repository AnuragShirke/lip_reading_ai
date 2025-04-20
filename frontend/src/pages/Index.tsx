
import React, { useState, useCallback } from "react";
import { VideoUploader } from "@/components/VideoUploader";
import { VideoPlayer } from "@/components/VideoPlayer";
import { PredictionDisplay } from "@/components/PredictionDisplay";
import { ResultsHistory, HistoryItem } from "@/components/ResultsHistory";
import { ThemeToggle } from "@/components/ThemeToggle";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { toast } from "sonner";
import { v4 as uuidv4 } from "uuid";
import { uploadVideo } from "@/services/api";

// Get API URL from environment or use default
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const Index = () => {
  // State
  const [currentVideo, setCurrentVideo] = useState<string | null>(null);
  const [currentVideoName, setCurrentVideoName] = useState<string>("");
  const [currentThumbnail, setCurrentThumbnail] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingStage, setProcessingStage] = useState<string>("");
  const [prediction, setPrediction] = useState<string | null>(null);
  const [confidence, setConfidence] = useState(0);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [selectedHistoryId, setSelectedHistoryId] = useState<string>("");
  const [activeTab, setActiveTab] = useState<string>("upload");

  // Handle file upload
  const handleVideoUpload = useCallback(async (file: File) => {
    try {
      // Create object URL for the video
      const videoUrl = URL.createObjectURL(file);
      setCurrentVideo(videoUrl);
      setCurrentVideoName(file.name);
      setPrediction(null);
      setConfidence(0);
      setIsProcessing(true);

      // Switch to video tab to show the preview
      setActiveTab("video");

      // Set processing stages
      setProcessingStage("Detecting facial landmarks...");
      await new Promise(resolve => setTimeout(resolve, 1000));

      setProcessingStage("Identifying lip region...");
      await new Promise(resolve => setTimeout(resolve, 1000));

      setProcessingStage("Analyzing lip movements...");
      await new Promise(resolve => setTimeout(resolve, 1000));

      setProcessingStage("Generating prediction...");

      // Make API call to backend for prediction
      try {
        console.log('Calling uploadVideo with file:', file.name, file.type, file.size);
        const result = await uploadVideo(file);
        console.log('Received result from backend:', result);

        // Update state with results
        setPrediction(result.prediction);
        setConfidence(result.confidence);
        setIsProcessing(false);
        setProcessingStage("");

        // If the backend returns a video URL, use it instead of the local object URL
        if (result.video_url) {
          // Create a full URL by combining the API URL with the video URL path
          const fullVideoUrl = `${API_URL}${result.video_url}`;
          console.log('Using video URL from backend:', fullVideoUrl);
          setCurrentVideo(fullVideoUrl);
        }

        // If the backend returns a thumbnail URL, use it
        if (result.thumbnail_url) {
          const fullThumbnailUrl = `${API_URL}${result.thumbnail_url}`;
          console.log('Using thumbnail URL from backend:', fullThumbnailUrl);
          setCurrentThumbnail(fullThumbnailUrl);
        } else {
          setCurrentThumbnail(null);
        }

        // Add to history
        const historyItem = {
          id: uuidv4(),
          timestamp: new Date(),
          videoName: file.name,
          prediction: result.prediction,
          confidence: result.confidence,
          videoUrl: result.video_url ? `${API_URL}${result.video_url}` : videoUrl, // Use backend URL if available
          thumbnailUrl: result.thumbnail_url ? `${API_URL}${result.thumbnail_url}` : null // Include thumbnail URL if available
        };

        setHistory(prev => [historyItem, ...prev]);
        setSelectedHistoryId(historyItem.id);

        // Show appropriate success message based on file type
        if (file.name.toLowerCase().endsWith('.mpg') || file.name.toLowerCase().endsWith('.mpeg')) {
          toast.success("Prediction complete! Note: MPG videos may not play directly in the browser, but you can download them.");
        } else {
          toast.success("Prediction complete");
        }
      } catch (error) {
        console.error('Error in handleVideoUpload:', error);
        setIsProcessing(false);
        setProcessingStage("");
        toast.error(`Error: ${error instanceof Error ? error.message : 'Failed to process video'}`);
      }
    } catch (error) {
      console.error('Error processing video:', error);
      setIsProcessing(false);
      setProcessingStage("");
      toast.error(error instanceof Error ? error.message : 'Failed to process video');
    }
  }, []);

  // Handle history item selection
  const handleHistorySelect = useCallback((item: HistoryItem) => {
    setSelectedHistoryId(item.id);
    setPrediction(item.prediction);
    setConfidence(item.confidence || 0);

    // Load the video if available in the history item
    if (item.videoUrl) {
      setCurrentVideo(item.videoUrl);
      setCurrentVideoName(item.videoName);

      // Set the thumbnail if available
      if (item.thumbnailUrl) {
        setCurrentThumbnail(item.thumbnailUrl);
      } else {
        setCurrentThumbnail(null);
      }

      setActiveTab("video");
    }
  }, []);

  return (
    <div className="min-h-screen bg-background flex flex-col">
      {/* Header */}
      <header className="border-b py-4">
        <div className="container flex justify-between items-start">
          <div>
            <h1 className="text-2xl font-medium">Lip Reading AI</h1>
            <p className="text-muted-foreground">Upload videos of lip movements and get real-time text predictions</p>
          </div>
          <ThemeToggle className="mt-1" />
        </div>
      </header>

      {/* Main content */}
      <main className="container flex-1 py-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Left column - Video upload & preview */}
          <div className="md:col-span-2 space-y-6">
            <Tabs value={activeTab} onValueChange={setActiveTab}>
              <TabsList>
                <TabsTrigger value="upload">Upload</TabsTrigger>
                <TabsTrigger value="video" disabled={!currentVideo}>Preview</TabsTrigger>
              </TabsList>
              <TabsContent value="upload" className="space-y-4 animate-fade-in">
                <VideoUploader
                  onVideoUpload={handleVideoUpload}
                  isUploading={isProcessing}
                />
              </TabsContent>
              <TabsContent value="video" className="space-y-4 animate-fade-in">
                <VideoPlayer videoSrc={currentVideo} thumbnailSrc={currentThumbnail} />
              </TabsContent>
            </Tabs>

            {/* Prediction Display */}
            <PredictionDisplay
              prediction={prediction}
              isProcessing={isProcessing}
              processingStage={processingStage}
              confidence={confidence}
            />
          </div>

          {/* Right column - History */}
          <div className="md:col-span-1">
            <ResultsHistory
              history={history}
              onSelect={handleHistorySelect}
              selectedId={selectedHistoryId}
              className="h-full"
            />
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t py-4">
        <div className="container text-center text-muted-foreground text-sm">
          <p>Lip Reading AI Application</p>
        </div>
      </footer>
    </div>
  );
};

export default Index;
