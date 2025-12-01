"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { useToast } from "@/hooks/use-toast"
import { Loader2, Play, TrendingUp, Award, BarChart3, Target } from "lucide-react"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Progress } from "@/components/ui/progress"

interface EvalResults {
  auc?: number
  seed_score?: number
  panel_mean?: number
  panel_max?: number
  improvement?: number
}

export function EvaluatePanel() {
  const { toast } = useToast()
  const [evaluating, setEvaluating] = useState(false)
  const [results, setResults] = useState<EvalResults | null>(null)

  const runEvaluation = async () => {
    console.log("[v0] Starting evaluation...")
    setEvaluating(true)
    setResults(null)

    try {
      const res = await fetch("/api/evaluate", { method: "POST" })
      const data = await res.json()

      console.log("[v0] Evaluation API response:", data)
      console.log("[v0] Response status:", res.ok)

      if (!res.ok) throw new Error(data.error || "Evaluation failed")

      console.log("[v0] Setting results:", data)
      setResults(data)

      if (typeof sessionStorage !== "undefined") {
        sessionStorage.setItem("abyss_eval_results", JSON.stringify(data))
        console.log("[v0] Saved results to sessionStorage")
      }

      toast({
        title: "Success",
        description: "Evaluation completed successfully",
      })
    } catch (error) {
      console.error("[v0] Evaluation error:", error)
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to run evaluation",
        variant: "destructive",
      })
    } finally {
      setEvaluating(false)
    }
  }

  useEffect(() => {
    if (typeof sessionStorage !== "undefined") {
      const stored = sessionStorage.getItem("abyss_eval_results")
      if (stored) {
        const parsedResults = JSON.parse(stored)
        console.log("[v0] Loaded results from sessionStorage:", parsedResults)
        setResults(parsedResults)
      }
    }
  }, [])

  console.log("[v0] Current results state:", results)

  return (
    <div className="space-y-6">
      <Card className="border-border bg-background">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5 text-blue-500" />
            Model Evaluation
          </CardTitle>
          <CardDescription>Evaluate model performance and compare seed vs panel</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <Alert>
            <AlertDescription>
              Evaluation includes global AUC on test data and comparison between seed CDR3 and generated panel.
            </AlertDescription>
          </Alert>

          <Button onClick={runEvaluation} disabled={evaluating} className="w-full h-12" size="lg">
            {evaluating ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Evaluating...
              </>
            ) : (
              <>
                <Play className="mr-2 h-4 w-4" />
                Run Evaluation
              </>
            )}
          </Button>
        </CardContent>
      </Card>

      {results && (
        <>
          {/* Global Performance */}
          {results.auc !== undefined && (
            <Card className="border-border bg-background">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Award className="h-5 w-5 text-blue-500" />
                  Global Performance
                </CardTitle>
                <CardDescription>Model AUC on test data</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="text-center p-6 rounded-lg border border-blue-500/50 bg-blue-950/20">
                    <div className="text-5xl font-bold text-blue-500 mb-2">{results.auc.toFixed(4)}</div>
                    <p className="text-sm text-muted-foreground">Area Under Curve (AUC)</p>
                  </div>

                  {/* Visual progress bar representation */}
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Performance</span>
                      <span className="font-medium">{(results.auc * 100).toFixed(1)}%</span>
                    </div>
                    <Progress value={results.auc * 100} className="h-3" />
                  </div>

                  <div className="grid grid-cols-3 gap-2 text-center text-xs">
                    <div>
                      <div className="text-muted-foreground">Poor</div>
                      <div className="font-mono">0.5</div>
                    </div>
                    <div>
                      <div className="text-muted-foreground">Good</div>
                      <div className="font-mono">0.75</div>
                    </div>
                    <div>
                      <div className="text-muted-foreground">Excellent</div>
                      <div className="font-mono">0.9+</div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Seed vs Panel Comparison */}
          {results.seed_score !== undefined && (
            <Card className="border-border bg-background">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Target className="h-5 w-5 text-blue-500" />
                  Seed vs Panel Comparison
                </CardTitle>
                <CardDescription>Binding score analysis and improvement metrics</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* Score comparison bars */}
                <div className="space-y-4">
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">Seed CDR3</span>
                      <span className="text-lg font-bold">{results.seed_score.toFixed(4)}</span>
                    </div>
                    <div className="relative h-8 rounded-lg border border-border bg-muted/50 overflow-hidden">
                      <div
                        className="absolute inset-y-0 left-0 bg-blue-500/30 transition-all"
                        style={{ width: `${(results.seed_score / (results.panel_max || 1)) * 100}%` }}
                      />
                      <div className="absolute inset-0 flex items-center justify-center text-xs font-medium">
                        Baseline
                      </div>
                    </div>
                  </div>

                  {results.panel_mean !== undefined && (
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium">Panel Mean</span>
                        <span className="text-lg font-bold">{results.panel_mean.toFixed(4)}</span>
                      </div>
                      <div className="relative h-8 rounded-lg border border-border bg-muted/50 overflow-hidden">
                        <div
                          className="absolute inset-y-0 left-0 bg-green-500/30 transition-all"
                          style={{ width: `${(results.panel_mean / (results.panel_max || 1)) * 100}%` }}
                        />
                        <div className="absolute inset-0 flex items-center justify-center text-xs font-medium">
                          Average
                        </div>
                      </div>
                    </div>
                  )}

                  {results.panel_max !== undefined && (
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium">Panel Maximum</span>
                        <span className="text-lg font-bold text-green-500">{results.panel_max.toFixed(4)}</span>
                      </div>
                      <div className="relative h-8 rounded-lg border border-green-500/50 bg-muted/50 overflow-hidden">
                        <div
                          className="absolute inset-y-0 left-0 bg-green-500/50 transition-all"
                          style={{ width: "100%" }}
                        />
                        <div className="absolute inset-0 flex items-center justify-center text-xs font-medium">
                          Best Candidate
                        </div>
                      </div>
                    </div>
                  )}
                </div>

                {/* Improvement metric */}
                {results.improvement !== undefined && (
                  <div
                    className={`rounded-lg border p-6 text-center ${
                      results.improvement > 0
                        ? "border-green-500/50 bg-green-950/20"
                        : "border-red-500/50 bg-red-950/20"
                    }`}
                  >
                    <div className="flex items-center justify-center gap-2 mb-2">
                      <TrendingUp
                        className={`h-6 w-6 ${results.improvement > 0 ? "text-green-500" : "text-red-500"}`}
                      />
                      <span className="text-sm font-medium text-muted-foreground">Improvement over Seed</span>
                    </div>
                    <div
                      className={`text-4xl font-bold ${results.improvement > 0 ? "text-green-500" : "text-red-500"}`}
                    >
                      {results.improvement > 0 ? "+" : ""}
                      {(results.improvement * 100).toFixed(2)}%
                    </div>
                    <p className="mt-2 text-xs text-muted-foreground">
                      {results.improvement > 0
                        ? "Panel successfully improved binding affinity"
                        : "Panel did not improve over seed"}
                    </p>
                  </div>
                )}

                {/* Summary stats */}
                <div className="grid grid-cols-2 gap-3">
                  <div className="rounded-lg border border-border bg-muted/50 p-4">
                    <p className="text-xs text-muted-foreground mb-1">Score Range</p>
                    <p className="text-sm font-mono font-medium">
                      {results.seed_score.toFixed(4)} → {results.panel_max?.toFixed(4)}
                    </p>
                  </div>
                  <div className="rounded-lg border border-border bg-muted/50 p-4">
                    <p className="text-xs text-muted-foreground mb-1">Absolute Gain</p>
                    <p className="text-sm font-mono font-medium">
                      +{((results.panel_max || 0) - results.seed_score).toFixed(4)}
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </>
      )}
    </div>
  )
}
