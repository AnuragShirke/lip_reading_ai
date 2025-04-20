
import React from 'react';
import { cn } from '@/lib/utils';
import { Badge } from '@/components/ui/badge';

interface TranscriptionBadgeProps {
  confidence: number;
  className?: string;
}

export function TranscriptionBadge({ confidence, className }: TranscriptionBadgeProps) {
  // Function to determine variant based on confidence
  const getVariant = () => {
    if (confidence >= 0.8) return "success";
    if (confidence >= 0.5) return "warning";
    return "destructive";
  };
  
  // Function to get label text based on confidence
  const getLabel = () => {
    if (confidence >= 0.8) return "High Confidence";
    if (confidence >= 0.5) return "Medium Confidence";
    return "Low Confidence";
  };
  
  // Custom color styles based on confidence
  const getStyles = () => {
    if (confidence >= 0.8) return "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-100";
    if (confidence >= 0.5) return "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-100";
    return "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-100";
  };
  
  return (
    <Badge 
      variant="outline"
      className={cn(getStyles(), "font-normal", className)}
    >
      {getLabel()}
    </Badge>
  );
}
