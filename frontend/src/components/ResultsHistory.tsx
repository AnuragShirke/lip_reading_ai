
import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Separator } from '@/components/ui/separator';
import { History } from 'lucide-react';
import { cn } from '@/lib/utils';

export interface HistoryItem {
  id: string;
  timestamp: Date;
  videoName: string;
  prediction: string;
  confidence?: number;
  videoUrl?: string; // URL to the video file
  thumbnailUrl?: string; // URL to the video thumbnail
}

interface ResultsHistoryProps {
  history: HistoryItem[];
  onSelect: (item: HistoryItem) => void;
  selectedId?: string;
  className?: string;
}

export function ResultsHistory({ history, onSelect, selectedId, className }: ResultsHistoryProps) {
  if (history.length === 0) {
    return (
      <Card className={cn("h-full", className)}>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <History className="h-5 w-5" />
            History
          </CardTitle>
        </CardHeader>
        <CardContent className="flex items-center justify-center h-[300px]">
          <div className="flex flex-col items-center justify-center text-center">
            <History className="h-6 w-6 mb-2 text-muted-foreground/50" />
            <p className="text-muted-foreground text-sm">No predictions yet</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className={cn("h-full", className)}>
      <CardHeader>
        <CardTitle className="text-lg flex items-center gap-2">
          <History className="h-5 w-5" />
          History
        </CardTitle>
      </CardHeader>
      <CardContent className="px-2">
        <div className="space-y-1 max-h-[350px] overflow-y-auto pr-2">
          {history.map((item, index) => (
            <React.Fragment key={item.id}>
              <Button
                variant={selectedId === item.id ? "secondary" : "ghost"}
                className={cn(
                  "w-full justify-start px-2 py-1 h-auto text-left",
                  selectedId === item.id && "bg-secondary"
                )}
                onClick={() => onSelect(item)}
              >
                <div className="flex flex-col gap-1 w-full">
                  <div className="flex justify-between items-center w-full">
                    <span className="font-medium truncate">{item.prediction}</span>
                    <span className="text-xs text-muted-foreground">
                      {item.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </span>
                  </div>
                  <div className="flex justify-between items-center w-full">
                    <span className="text-xs text-muted-foreground truncate max-w-[180px]">
                      {item.videoName}
                    </span>
                    {item.confidence !== undefined && (
                      <span className="text-xs">
                        {(item.confidence * 100).toFixed(0)}%
                      </span>
                    )}
                  </div>
                </div>
              </Button>
              {index < history.length - 1 && <Separator className="my-1" />}
            </React.Fragment>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
