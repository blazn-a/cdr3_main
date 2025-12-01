"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { useToast } from "@/hooks/use-toast"
import { Download, FileText, Loader2, Database, TrendingUp, Layers } from "lucide-react"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"

interface Candidate {
  cdr3: string
  score: number
  cluster?: number
}

export function ResultsPanel() {
  const { toast } = useToast()
  const [loading, setLoading] = useState(true)
  const [results, setResults] = useState<{
    raw?: string
    filtered?: string
    panel?: string
    candidates?: Candidate[]
  }>({})

  useEffect(() => {
    loadResults()
  }, [])

  const loadResults = async () => {
    try {
      if (typeof sessionStorage !== "undefined") {
        const rawDataStr = sessionStorage.getItem("abyss_last_raw")
        const filteredDataStr = sessionStorage.getItem("abyss_last_filtered")
        const panelDataStr = sessionStorage.getItem("abyss_last_panel")

        const loadedResults: any = {
          raw: rawDataStr ? "Generated" : null,
          filtered: filteredDataStr ? "Filtered" : null,
          panel: panelDataStr ? "Panel Created" : null,
          candidates: [],
        }

        // Load panel candidates for preview
        if (panelDataStr) {
          try {
            const panelData = JSON.parse(panelDataStr)
            if (panelData.candidates && Array.isArray(panelData.candidates)) {
              loadedResults.candidates = panelData.candidates.slice(0, 20).map((c: any) => ({
                cdr3: c.cdr3,
                score: c.score,
                cluster: c.cluster,
              }))
            }
          } catch (error) {
            console.log("[v0] Could not load candidates preview:", error)
          }
        }

        setResults(loadedResults)
      }
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to load results",
        variant: "destructive",
      })
    } finally {
      setLoading(false)
    }
  }

  const downloadFile = async (type: "raw" | "filtered" | "panel" | "fasta") => {
    try {
      const res = await fetch(`/api/download/${type}`)
      if (!res.ok) throw new Error("File not found")

      const blob = await res.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement("a")
      a.href = url
      a.download = `${type}.${type === "fasta" ? "fasta" : "csv"}`
      document.body.appendChild(a)
      a.click()
      window.URL.revokeObjectURL(url)
      document.body.removeChild(a)

      toast({
        title: "Success",
        description: "File downloaded successfully",
      })
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to download file",
        variant: "destructive",
      })
    }
  }

  if (loading) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center py-12">
          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        </CardContent>
      </Card>
    )
  }

  const stats = {
    totalCandidates: results.candidates?.length || 0,
    avgScore: results.candidates?.length
      ? (results.candidates.reduce((sum, c) => sum + c.score, 0) / results.candidates.length).toFixed(4)
      : "0.0000",
    uniqueClusters: results.candidates
      ? new Set(results.candidates.map((c) => c.cluster).filter((c) => c !== undefined)).size
      : 0,
  }

  return (
    <div className="space-y-6">
      {results.candidates && results.candidates.length > 0 && (
        <div className="grid gap-4 md:grid-cols-3">
          <Card className="border-border">
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center gap-2 text-sm font-medium text-muted-foreground">
                <Database className="h-4 w-4" />
                Total Candidates
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stats.totalCandidates}</div>
              <p className="text-xs text-muted-foreground">Generated sequences</p>
            </CardContent>
          </Card>

          <Card className="border-border">
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center gap-2 text-sm font-medium text-muted-foreground">
                <TrendingUp className="h-4 w-4" />
                Average Score
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stats.avgScore}</div>
              <p className="text-xs text-muted-foreground">Binding affinity prediction</p>
            </CardContent>
          </Card>

          <Card className="border-border">
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center gap-2 text-sm font-medium text-muted-foreground">
                <Layers className="h-4 w-4" />
                Unique Clusters
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stats.uniqueClusters}</div>
              <p className="text-xs text-muted-foreground">Diverse binding modes</p>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Output Files */}
      <Card>
        <CardHeader>
          <CardTitle>Output Files</CardTitle>
          <CardDescription>Download generated results in various formats</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex flex-col gap-3 rounded-lg border border-border p-4 sm:flex-row sm:items-center sm:justify-between">
            <div className="flex items-start gap-3">
              <div className="rounded-lg bg-blue-500/10 p-2">
                <Database className="h-5 w-5 text-blue-500" />
              </div>
              <div>
                <p className="font-medium">Raw Candidates</p>
                <p className="text-sm text-muted-foreground">{results.raw || "Not generated yet"}</p>
              </div>
            </div>
            <Button onClick={() => downloadFile("raw")} disabled={!results.raw} variant="outline" size="sm">
              <Download className="mr-2 h-4 w-4" />
              Download CSV
            </Button>
          </div>

          <div className="flex flex-col gap-3 rounded-lg border border-border p-4 sm:flex-row sm:items-center sm:justify-between">
            <div className="flex items-start gap-3">
              <div className="rounded-lg bg-blue-500/10 p-2">
                <TrendingUp className="h-5 w-5 text-blue-500" />
              </div>
              <div>
                <p className="font-medium">Filtered Candidates</p>
                <p className="text-sm text-muted-foreground">{results.filtered || "Not generated yet"}</p>
              </div>
            </div>
            <Button onClick={() => downloadFile("filtered")} disabled={!results.filtered} variant="outline" size="sm">
              <Download className="mr-2 h-4 w-4" />
              Download CSV
            </Button>
          </div>

          <div className="flex flex-col gap-3 rounded-lg border border-border p-4 sm:flex-row sm:items-center sm:justify-between">
            <div className="flex items-start gap-3">
              <div className="rounded-lg bg-blue-500/10 p-2">
                <Layers className="h-5 w-5 text-blue-500" />
              </div>
              <div>
                <p className="font-medium">Final Panel (CSV)</p>
                <p className="text-sm text-muted-foreground">{results.panel || "Not generated yet"}</p>
              </div>
            </div>
            <Button onClick={() => downloadFile("panel")} disabled={!results.panel} variant="outline" size="sm">
              <Download className="mr-2 h-4 w-4" />
              Download CSV
            </Button>
          </div>

          <div className="flex flex-col gap-3 rounded-lg border border-border p-4 sm:flex-row sm:items-center sm:justify-between">
            <div className="flex items-start gap-3">
              <div className="rounded-lg bg-blue-500/10 p-2">
                <FileText className="h-5 w-5 text-blue-500" />
              </div>
              <div>
                <p className="font-medium">Final Panel (FASTA)</p>
                <p className="text-sm text-muted-foreground">Sequences in FASTA format for downstream analysis</p>
              </div>
            </div>
            <Button onClick={() => downloadFile("fasta")} disabled={!results.panel} variant="outline" size="sm">
              <FileText className="mr-2 h-4 w-4" />
              Download FASTA
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Candidates Table */}
      {results.candidates && results.candidates.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Top Candidates</CardTitle>
            <CardDescription>Preview of generated CDR3 sequences with binding scores</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="rounded-md border">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-16">#</TableHead>
                    <TableHead>CDR3 Sequence</TableHead>
                    <TableHead className="w-32 text-right">Score</TableHead>
                    {results.candidates.some((c) => c.cluster !== undefined) && (
                      <TableHead className="w-24 text-right">Cluster</TableHead>
                    )}
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {results.candidates.map((candidate, idx) => (
                    <TableRow key={idx}>
                      <TableCell className="font-medium">{idx + 1}</TableCell>
                      <TableCell className="font-mono text-sm">{candidate.cdr3}</TableCell>
                      <TableCell className="text-right">
                        <Badge variant={candidate.score > 0.8 ? "default" : "secondary"}>
                          {candidate.score.toFixed(4)}
                        </Badge>
                      </TableCell>
                      {candidate.cluster !== undefined && (
                        <TableCell className="text-right">
                          <Badge variant="outline">{candidate.cluster}</Badge>
                        </TableCell>
                      )}
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
