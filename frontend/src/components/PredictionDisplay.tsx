
import React, { useState, useEffect, useRef } from 'react';
import { cn } from '@/lib/utils';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Loader, FileText } from 'lucide-react';

interface PredictionDisplayProps {
  prediction: string | null;
  isProcessing: boolean;
  processingStage?: string;
  confidence?: number;
  className?: string;
}

export function PredictionDisplay({ prediction, isProcessing, processingStage = "", confidence = 0, className }: PredictionDisplayProps) {
  const [displayText, setDisplayText] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const previousPrediction = useRef<string | null>(null);

  // Typing animation effect
  useEffect(() => {
    if (prediction === previousPrediction.current) return;
    previousPrediction.current = prediction;

    if (!prediction) {
      setDisplayText('');
      return;
    }

    setIsTyping(true);
    setDisplayText('');

    let index = 0;
    const timer = setInterval(() => {
      if (index < prediction.length) {
        setDisplayText((prev) => prev + prediction.charAt(index));
        index++;
      } else {
        clearInterval(timer);
        setIsTyping(false);
      }
    }, 50); // Speed of typing animation

    return () => clearInterval(timer);
  }, [prediction]);

  return (
    <Card className={cn("overflow-hidden", className)}>
      <CardContent className="p-6 flex flex-col">
        <div className="flex justify-between items-center mb-2">
          <h3 className="text-lg font-medium">Prediction</h3>
          {confidence > 0 && !isProcessing && (
            <Badge variant="outline" className={cn(
              confidence > 0.8 ? "bg-green-100 text-green-800" : 
              confidence > 0.5 ? "bg-yellow-100 text-yellow-800" : 
              "bg-red-100 text-red-800"
            )}>
              {confidence > 0.8 ? "High" : confidence > 0.5 ? "Medium" : "Low"} Confidence: {(confidence * 100).toFixed(0)}%
            </Badge>
          )}
        </div>
        
        <div className="h-32 flex items-center justify-center">
          {isProcessing ? (
            <div className="flex flex-col items-center space-y-2">
              <Loader className="h-8 w-8 animate-spin text-lip" />
              <p className="text-sm text-muted-foreground animate-pulse">
                {processingStage || "Processing video..."}
              </p>
            </div>
          ) : displayText ? (
            <div className="relative">
              <p className="text-3xl font-medium text-foreground">
                {displayText}
                {isTyping && (
                  <span className="inline-block w-[3px] h-[1.2em] bg-lip ml-1 align-middle animate-blink"></span>
                )}
              </p>
              <div className="absolute -bottom-4 left-0 right-0 h-[2px] bg-gradient-to-r from-transparent via-lip to-transparent opacity-30" />
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center text-center">
              <FileText className="h-6 w-6 mb-2 text-muted-foreground/50" />
              <p className="text-muted-foreground text-sm">
                Upload a video to see prediction
              </p>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
