
import React from 'react';
import { Loader, Text } from 'lucide-react';
import { cn } from '@/lib/utils';

interface ProcessingIndicatorProps {
  text?: string;
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

export function ProcessingIndicator({ 
  text = "Processing...", 
  size = 'md',
  className 
}: ProcessingIndicatorProps) {
  const sizeClasses = {
    sm: 'h-4 w-4',
    md: 'h-6 w-6',
    lg: 'h-8 w-8'
  };
  
  const textClasses = {
    sm: 'text-xs',
    md: 'text-sm',
    lg: 'text-base'
  };
  
  return (
    <div className={cn("flex flex-col items-center justify-center gap-2", className)}>
      <Loader className={cn("animate-spin text-lip", sizeClasses[size])} />
      {text && <p className={cn("text-muted-foreground animate-pulse", textClasses[size])}>{text}</p>}
    </div>
  );
}
