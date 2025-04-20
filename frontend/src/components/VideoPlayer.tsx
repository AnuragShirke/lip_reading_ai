
import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, Square, ArrowLeft, ArrowRight } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { cn } from '@/lib/utils';

interface VideoPlayerProps {
  videoSrc: string | null;
  thumbnailSrc?: string | null;
  className?: string;
}

export function VideoPlayer({ videoSrc, thumbnailSrc, className }: VideoPlayerProps) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [showLipDetection, setShowLipDetection] = useState(true);
  const videoRef = useRef<HTMLVideoElement>(null);

  // State to track if video is loaded and ready
  const [isVideoReady, setIsVideoReady] = useState(false);
  const [videoError, setVideoError] = useState<string | null>(null);

  // Reset player state when video changes
  useEffect(() => {
    if (videoRef.current && videoSrc) {
      videoRef.current.currentTime = 0;
      setCurrentTime(0);
      setIsPlaying(false);
      setIsVideoReady(false);
      setVideoError(null);
    }
  }, [videoSrc]);

  // Handle video loaded event
  const handleVideoLoaded = () => {
    setIsVideoReady(true);
    setVideoError(null);

    // Try to auto-play once the video is loaded
    if (videoRef.current) {
      videoRef.current.play()
        .then(() => setIsPlaying(true))
        .catch(err => {
          console.error('Auto-play failed:', err);
          setIsPlaying(false);
          // Don't set error for autoplay policy issues
          if (!(err.name === 'NotAllowedError')) {
            setVideoError(`Playback error: ${err.message}`);
          }
        });
    }
  };

  // Handle video error event
  const handleVideoError = () => {
    setVideoError('This video format may not be supported by your browser');
    setIsVideoReady(false);
    setIsPlaying(false);
  };

  // Update current time as video plays
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const updateTime = () => setCurrentTime(video.currentTime);
    const updateDuration = () => setDuration(video.duration);
    const handleEnd = () => setIsPlaying(false);

    video.addEventListener('timeupdate', updateTime);
    video.addEventListener('durationchange', updateDuration);
    video.addEventListener('ended', handleEnd);

    return () => {
      video.removeEventListener('timeupdate', updateTime);
      video.removeEventListener('durationchange', updateDuration);
      video.removeEventListener('ended', handleEnd);
    };
  }, []);

  // Play/Pause logic
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    if (isPlaying) {
      video.play().catch(() => setIsPlaying(false));
    } else {
      video.pause();
    }
  }, [isPlaying]);

  // Handle seeking with frame controls (assuming ~30fps)
  const seekFrame = (forward: boolean) => {
    const video = videoRef.current;
    if (!video) return;

    const frameTime = 1/30; // Approx. 33ms for 30fps
    const newTime = forward
      ? Math.min(video.duration, currentTime + frameTime)
      : Math.max(0, currentTime - frameTime);

    video.currentTime = newTime;
    setCurrentTime(newTime);
  };

  // Format time display (mm:ss)
  const formatTime = (time: number) => {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
  };

  // Handle slider change
  const handleSliderChange = (value: number[]) => {
    const newTime = value[0];
    if (videoRef.current) {
      videoRef.current.currentTime = newTime;
      setCurrentTime(newTime);
    }
  };

  return (
    <div className={cn("flex flex-col space-y-2", className)}>
      <div className="relative w-full aspect-video bg-secondary/50 rounded-lg overflow-hidden">
        {videoSrc ? (
          <>
            <div className="relative w-full h-full">
              <video
                ref={videoRef}
                className="w-full h-full object-contain"
                loop
                playsInline
                onLoadedData={handleVideoLoaded}
                onError={handleVideoError}
              >
                {/* Support multiple video formats */}
                <source src={videoSrc} type="video/mp4" />
                <source src={videoSrc} type="video/webm" />
                <source src={videoSrc} type="video/ogg" />
                <source src={videoSrc} type="video/mpeg" />
                <source src={videoSrc} type="video/mpg" />
                Your browser does not support the video tag.
              </video>

              {/* Show error message if video fails to load */}
              {videoError && (
                <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/70 text-white p-4 text-center">
                  {/* Show thumbnail if available */}
                  {thumbnailSrc && (
                    <div className="mb-4 max-w-[80%] max-h-[50%] overflow-hidden rounded-md">
                      <img
                        src={thumbnailSrc}
                        alt="Video thumbnail"
                        className="w-full h-auto object-contain"
                      />
                    </div>
                  )}
                  <div>
                    <p className="mb-2">{videoError}</p>
                    <p className="text-sm mb-3">
                      {videoSrc && videoSrc.includes('.mpg') ?
                        "MPG videos can't be played directly in most browsers. The prediction has been processed successfully." :
                        "This video format may not be supported by your browser."}
                    </p>
                    <p className="text-sm mb-3">Download the video to view it in a media player like VLC.</p>
                    <a
                      href={videoSrc}
                      download
                      className="px-3 py-1 bg-primary text-primary-foreground rounded-md text-sm hover:bg-primary/90"
                    >
                      Download Video
                    </a>
                  </div>
                </div>
              )}
            </div>
            {showLipDetection && (
              <div className="absolute top-1/3 left-1/4 w-1/2 h-1/3 border-2 border-lip/70 rounded-lg pointer-events-none">
                <div className="absolute top-[-8px] left-[-8px] right-[-8px] bottom-[-8px] border border-lip/30 rounded-lg animate-pulse" />
              </div>
            )}
          </>
        ) : (
          <div className="w-full h-full flex items-center justify-center text-muted-foreground">
            Upload a video to preview
          </div>
        )}
      </div>

      {videoSrc && !videoError && (
        <div className="space-y-2">
          <div className="flex items-center space-x-2">
            <Slider
              value={[currentTime]}
              max={duration || 100}
              step={0.01}
              onValueChange={handleSliderChange}
              disabled={!videoSrc}
              className="flex-1"
            />
            <span className="text-xs tabular-nums text-muted-foreground w-14 text-right">
              {formatTime(currentTime)} / {formatTime(duration || 0)}
            </span>
          </div>

          <div className="flex items-center justify-between">
            <div className="flex space-x-1">
              <Button
                size="icon"
                variant="outline"
                onClick={() => seekFrame(false)}
                disabled={!videoSrc || currentTime <= 0}
              >
                <ArrowLeft className="h-4 w-4" />
              </Button>
              <Button
                size="icon"
                variant="outline"
                onClick={() => setIsPlaying(!isPlaying)}
                disabled={!videoSrc}
              >
                {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
              </Button>
              <Button
                size="icon"
                variant="outline"
                onClick={() => seekFrame(true)}
                disabled={!videoSrc || currentTime >= duration}
              >
                <ArrowRight className="h-4 w-4" />
              </Button>
            </div>
            <Button
              size="sm"
              variant="ghost"
              onClick={() => setShowLipDetection(!showLipDetection)}
              disabled={!videoSrc}
              className="text-xs"
            >
              {showLipDetection ? "Hide" : "Show"} Lip Detection
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
