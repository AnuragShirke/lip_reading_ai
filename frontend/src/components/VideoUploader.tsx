
import React, { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, File } from "lucide-react";
import { cn } from "@/lib/utils";
import { toast } from "sonner";

interface VideoUploaderProps {
  onVideoUpload: (file: File) => void;
  isUploading: boolean;
}

export function VideoUploader({ onVideoUpload, isUploading }: VideoUploaderProps) {
  const [isDragging, setIsDragging] = useState(false);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) return;

    const file = acceptedFiles[0];
    const fileName = file.name.toLowerCase();

    // Check if it's a video file either by MIME type or extension
    if (!file.type.startsWith("video/") &&
        !fileName.endsWith('.mp4') &&
        !fileName.endsWith('.mpg') &&
        !fileName.endsWith('.mpeg') &&
        !fileName.endsWith('.avi') &&
        !fileName.endsWith('.mov') &&
        !fileName.endsWith('.webm')) {
      toast.error("Please upload a valid video file (MP4, MPG, AVI, MOV, WebM)");
      return;
    }

    // For MPG files, show a note about conversion
    if (fileName.endsWith('.mpg') || fileName.endsWith('.mpeg')) {
      toast.info("MPG file detected. It will be converted to MP4 for playback.");
    }

    onVideoUpload(file);
  }, [onVideoUpload]);

  const { getRootProps, getInputProps } = useDropzone({
    onDrop,
    accept: {
      "video/*": [],
      ".mp4": [],
      ".mpg": [],
      ".mpeg": [],
      ".avi": [],
      ".mov": [],
      ".webm": []
    },
    maxFiles: 1,
    onDragEnter: () => setIsDragging(true),
    onDragLeave: () => setIsDragging(false),
    disabled: isUploading
  });

  return (
    <div
      {...getRootProps()}
      className={cn(
        "border-2 border-dashed rounded-lg p-8 transition-all duration-300 flex flex-col items-center justify-center cursor-pointer min-h-[240px]",
        isDragging ? "border-lip bg-lip/5 scale-[1.01] shadow-sm" : "border-border hover:border-lip/50 hover:bg-secondary/30",
        isUploading && "opacity-50 pointer-events-none"
      )}
    >
      <input {...getInputProps()} />
      <div className="flex flex-col items-center gap-2 text-center">
        {isUploading ? (
          <div className="animate-spin-slow text-lip">
            <Upload size={36} />
          </div>
        ) : (
          <>
            <div className="p-4 bg-secondary rounded-full text-lip">
              <Upload size={24} />
            </div>
            <h3 className="text-lg font-medium mt-2">Drag video here</h3>
            <p className="text-sm text-muted-foreground mb-2">
              or click to browse
            </p>
            <p className="text-xs text-muted-foreground max-w-xs">
              Upload a short video clip of lip movements (MP4, MPG, WebM, MOV, AVI)
            </p>
          </>
        )}
      </div>
    </div>
  );
}
