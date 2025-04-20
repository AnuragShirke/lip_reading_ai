
import React from 'react';
import { cn } from '@/lib/utils';

interface LipDetectionOverlayProps {
  visible: boolean;
  className?: string;
}

export function LipDetectionOverlay({ visible, className }: LipDetectionOverlayProps) {
  if (!visible) return null;

  return (
    <div 
      className={cn(
        "absolute top-1/3 left-1/4 w-1/2 h-1/3 border-2 border-lip/70 rounded-lg pointer-events-none",
        "transition-all duration-200",
        "after:content-[''] after:absolute after:top-[-8px] after:left-[-8px] after:right-[-8px] after:bottom-[-8px] after:border after:border-lip/30 after:rounded-lg",
        className
      )}
    />
  );
}
