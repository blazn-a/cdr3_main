"use client"

import type React from "react"

import { useState, useRef } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { useToast } from "@/hooks/use-toast"
import { Loader2, Play, Database, Cpu, BarChart3, CheckCircle2, FileText, Upload, X } from "lucide-react"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"

export function TrainPanel() {
  const { toast } = useToast()
  const [training, setTraining] = useState(false)
  const [progress, setProgress] = useState(0)
  const [stage, setStage] = useState("")
  const [logs, setLogs] = useState<string[]>([])
  const [metrics, setMetrics] = useState<{ auc?: number; accuracy?: number } | null>(null)
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      if (!file.name.endsWith(".csv")) {
        toast({
          title: "Invalid file",
          description: "Please upload a CSV file",
          variant: "destructive",
        })
        return
      }
      setUploadedFile(file)
      toast({
        title: "File uploaded",
        description: `${file.name} ready for training`,
      })
    }
  }

  const runTraining = async () => {
    setTraining(true)
    setProgress(0)
    setLogs([])
    setMetrics(null)

    try {
      setStage("Embedding pairs with ESM...")
      setProgress(10)
      setLogs((prev) => [...prev, "Starting embedding process..."])

      const formData = new FormData()
      if (uploadedFile) {
        formData.append("file", uploadedFile)
        setLogs((prev) => [...prev, `Using uploaded file: ${uploadedFile.name}`])
      }

      const res = await fetch("/api/train", {
        method: "POST",
        body: uploadedFile ? formData : undefined,
      })
      const data = await res.json()

      if (!res.ok) throw new Error(data.error || "Training failed")

      setProgress(100)
      setStage("Training complete!")
      setLogs((prev) => [...prev, "Training completed successfully", `AUC: ${data.auc || "N/A"}`])
      setMetrics({ auc: data.auc, accuracy: data.accuracy })

      toast({
        title: "Success",
        description: "Model trained successfully",
      })
    } catch (error) {
      setStage("Training failed")
      setLogs((prev) => [...prev, `Error: ${error instanceof Error ? error.message : "Unknown error"}`])
      toast({
        title: "Error",
        description: "Failed to train model",
        variant: "destructive",
      })
    } finally {
      setTraining(false)
    }
  }

  return (
    <div className="space-y-6">
      <div className="grid gap-4 md:grid-cols-3">
        <Card className="border-border">
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-sm font-medium text-muted-foreground">
              <Database className="h-4 w-4" />
              Data Preparation
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-xs text-muted-foreground">
              Load antigen-CDR3 pairs and apply data augmentation with hard negatives and oversampling
            </p>
          </CardContent>
        </Card>

        <Card className="border-border">
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-sm font-medium text-muted-foreground">
              <Cpu className="h-4 w-4" />
              ESM Embedding
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-xs text-muted-foreground">
              Generate protein embeddings using ESM-2 backbone for sequence representation
            </p>
          </CardContent>
        </Card>

        <Card className="border-border">
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-sm font-medium text-muted-foreground">
              <BarChart3 className="h-4 w-4" />
              Model Training
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-xs text-muted-foreground">
              Train scoring model to predict binding affinity between antigens and CDR3 sequences
            </p>
          </CardContent>
        </Card>
      </div>

      <Card className="border-border bg-background">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Upload className="h-5 w-5 text-blue-500" />
            Upload Training Data
          </CardTitle>
          <CardDescription>Upload a CSV file with antigen-CDR3 pairs for training</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div
            className="border-2 border-dashed border-border rounded-lg p-8 text-center hover:border-blue-500 transition-colors cursor-pointer"
            onClick={() => fileInputRef.current?.click()}
          >
            <input ref={fileInputRef} type="file" accept=".csv" onChange={handleFileUpload} className="hidden" />
            {uploadedFile ? (
              <div className="flex items-center justify-center gap-3">
                <FileText className="h-8 w-8 text-blue-500" />
                <div className="text-left">
                  <p className="font-medium">{uploadedFile.name}</p>
                  <p className="text-sm text-muted-foreground">{(uploadedFile.size / 1024).toFixed(2)} KB</p>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={(e) => {
                    e.stopPropagation()
                    setUploadedFile(null)
                  }}
                >
                  <X className="h-4 w-4" />
                </Button>
              </div>
            ) : (
              <div>
                <Upload className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
                <p className="text-sm font-medium mb-1">Click to upload or drag and drop</p>
                <p className="text-xs text-muted-foreground">CSV file with columns: antigen, cdr3, label</p>
              </div>
            )}
          </div>

          <Alert>
            <Database className="h-4 w-4" />
            <AlertDescription>
              CSV format: <code className="rounded bg-muted px-1 py-0.5">antigen,cdr3,label</code>
              <br />
              Example: <code className="rounded bg-muted px-1 py-0.5 text-xs">SPIKE_PROTEIN,CARDGTYF,1</code>
            </AlertDescription>
          </Alert>
        </CardContent>
      </Card>

      <Card className="border-border bg-background">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Play className="h-5 w-5 text-blue-500" />
            Train Scoring Model
          </CardTitle>
          <CardDescription>Embed antigen-CDR3 pairs and train the scoring model</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <Button onClick={runTraining} disabled={training} className="w-full h-12" size="lg">
            {training ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Training...
              </>
            ) : (
              <>
                <Play className="mr-2 h-4 w-4" />
                Start Training
              </>
            )}
          </Button>

          {training && (
            <div className="space-y-2">
              <Progress value={progress} className="h-2" />
              <p className="text-sm text-muted-foreground">{stage}</p>
            </div>
          )}

          {metrics && (
            <div className="grid gap-3 sm:grid-cols-2">
              <div className="rounded-lg border border-green-500/50 bg-green-950/20 p-4">
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="h-4 w-4 text-green-500" />
                  <p className="text-sm font-medium">AUC Score</p>
                </div>
                <p className="mt-2 text-2xl font-bold">{metrics.auc?.toFixed(4) || "N/A"}</p>
              </div>
              {metrics.accuracy && (
                <div className="rounded-lg border border-green-500/50 bg-green-950/20 p-4">
                  <div className="flex items-center gap-2">
                    <CheckCircle2 className="h-4 w-4 text-green-500" />
                    <p className="text-sm font-medium">Accuracy</p>
                  </div>
                  <p className="mt-2 text-2xl font-bold">{(metrics.accuracy * 100).toFixed(2)}%</p>
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Training Logs */}
      {logs.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FileText className="h-5 w-5" />
              Training Logs
            </CardTitle>
            <CardDescription>Real-time training progress and output</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="rounded-md border border-border bg-muted/50 p-4">
              <pre className="text-xs text-muted-foreground">
                {logs.map((log, idx) => (
                  <div key={idx} className="py-0.5">
                    <Badge variant="outline" className="mr-2 font-mono text-[10px]">
                      {new Date().toLocaleTimeString()}
                    </Badge>
                    {log}
                  </div>
                ))}
              </pre>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
