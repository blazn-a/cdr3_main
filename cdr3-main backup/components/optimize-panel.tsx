"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { useToast } from "@/hooks/use-toast"
import { Loader2, Play, Filter, Grid3x3, Zap, CheckCircle2, Circle, ArrowRight } from "lucide-react"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"

type JobStatus = "idle" | "running" | "success" | "error"

export function OptimizePanel() {
  const { toast } = useToast()
  const [optimizeStatus, setOptimizeStatus] = useState<JobStatus>("idle")
  const [filterStatus, setFilterStatus] = useState<JobStatus>("idle")
  const [clusterStatus, setClusterStatus] = useState<JobStatus>("idle")
  const [quickStatus, setQuickStatus] = useState<JobStatus>("idle")
  const [progress, setProgress] = useState(0)
  const [message, setMessage] = useState("")
  const [stats, setStats] = useState({
    generated: 0,
    filtered: 0,
    clustered: 0,
  })

  const [pipelineData, setPipelineData] = useState<any>(null)

  useEffect(() => {
    if (typeof sessionStorage !== "undefined") {
      const savedState = sessionStorage.getItem("abyss_pipeline_state")
      if (savedState) {
        try {
          const state = JSON.parse(savedState)
          setOptimizeStatus(state.optimizeStatus || "idle")
          setFilterStatus(state.filterStatus || "idle")
          setClusterStatus(state.clusterStatus || "idle")
          setQuickStatus(state.quickStatus || "idle")
          setStats(state.stats || { generated: 0, filtered: 0, clustered: 0 })
        } catch (error) {
          console.log("[v0] Could not restore pipeline state")
        }
      }
    }
  }, [])

  useEffect(() => {
    if (typeof sessionStorage !== "undefined") {
      const state = {
        optimizeStatus,
        filterStatus,
        clusterStatus,
        quickStatus,
        stats,
      }
      sessionStorage.setItem("abyss_pipeline_state", JSON.stringify(state))
    }
  }, [optimizeStatus, filterStatus, clusterStatus, quickStatus, stats])

  const runOptimize = async () => {
    setOptimizeStatus("running")
    setProgress(0)
    setMessage("Generating CDR3 candidates...")

    try {
      const configStr = sessionStorage.getItem("abyss_config")
      const config = configStr ? JSON.parse(configStr) : {}

      const res = await fetch("/api/optimize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ config }),
      })
      const data = await res.json()

      if (!res.ok) throw new Error(data.error || "Optimization failed")

      setOptimizeStatus("success")
      setProgress(100)
      setStats((prev) => ({ ...prev, generated: data.count || 0 }))
      setMessage(`Generated ${data.count || 0} candidates`)

      setPipelineData(data.data)

      if (typeof sessionStorage !== "undefined") {
        sessionStorage.setItem("abyss_optimize_data", JSON.stringify(data.data))
        sessionStorage.setItem("abyss_last_raw", JSON.stringify(data.data))
      }

      toast({
        title: "Success",
        description: "CDR3 candidates generated successfully",
      })

      return data.data
    } catch (error) {
      setOptimizeStatus("error")
      setMessage(error instanceof Error ? error.message : "Optimization failed")
      toast({
        title: "Error",
        description: "Failed to generate candidates",
        variant: "destructive",
      })
      throw error
    }
  }

  const runFilter = async (inputData?: any) => {
    setFilterStatus("running")
    setProgress(0)
    setMessage("Filtering candidates...")

    try {
      let dataToFilter = inputData || pipelineData

      if (!dataToFilter && typeof sessionStorage !== "undefined") {
        const stored = sessionStorage.getItem("abyss_optimize_data")
        if (stored) {
          dataToFilter = JSON.parse(stored)
          setPipelineData(dataToFilter)
        }
      }

      if (!dataToFilter) {
        throw new Error("No data to filter. Run optimization first.")
      }

      const configStr = sessionStorage.getItem("abyss_config")
      const config = configStr ? JSON.parse(configStr) : {}

      const res = await fetch("/api/filter", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ data: dataToFilter, config }),
      })
      const data = await res.json()

      if (!res.ok) throw new Error(data.error || "Filtering failed")

      setFilterStatus("success")
      setProgress(100)
      setStats((prev) => ({ ...prev, filtered: data.count || 0 }))
      setMessage(`Filtered to ${data.count || 0} top candidates`)

      setPipelineData(data.data)

      if (typeof sessionStorage !== "undefined") {
        sessionStorage.setItem("abyss_filter_data", JSON.stringify(data.data))
        sessionStorage.setItem("abyss_last_filtered", JSON.stringify(data.data))
      }

      toast({
        title: "Success",
        description: "Candidates filtered successfully",
      })

      return data.data
    } catch (error) {
      setFilterStatus("error")
      setMessage(error instanceof Error ? error.message : "Filtering failed")
      toast({
        title: "Error",
        description: "Failed to filter candidates",
        variant: "destructive",
      })
      throw error
    }
  }

  const runCluster = async (inputData?: any) => {
    setClusterStatus("running")
    setProgress(0)
    setMessage("Clustering into panel...")

    try {
      let dataToCluster = inputData || pipelineData

      if (!dataToCluster && typeof sessionStorage !== "undefined") {
        const stored = sessionStorage.getItem("abyss_filter_data")
        if (stored) {
          dataToCluster = JSON.parse(stored)
          setPipelineData(dataToCluster)
        }
      }

      if (!dataToCluster) {
        throw new Error("No data to cluster. Run filtering first.")
      }

      const configStr = sessionStorage.getItem("abyss_config")
      const config = configStr ? JSON.parse(configStr) : {}

      const res = await fetch("/api/cluster", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ data: dataToCluster, config }),
      })
      const data = await res.json()

      if (!res.ok) throw new Error(data.error || "Clustering failed")

      setClusterStatus("success")
      setProgress(100)
      setStats((prev) => ({ ...prev, clustered: data.count || 0 }))
      setMessage(`Created panel with ${data.count || 0} clusters`)

      if (typeof sessionStorage !== "undefined") {
        sessionStorage.setItem("abyss_last_panel", JSON.stringify(data.data))
      }

      toast({
        title: "Success",
        description: "Panel created successfully",
      })

      return data.data
    } catch (error) {
      setClusterStatus("error")
      setMessage(error instanceof Error ? error.message : "Clustering failed")
      toast({
        title: "Error",
        description: "Failed to create panel",
        variant: "destructive",
      })
      throw error
    }
  }

  const runQuickPipeline = async () => {
    setQuickStatus("running")
    setProgress(0)

    try {
      setMessage("Step 1/3: Generating candidates...")
      setProgress(10)
      const optimizeData = await runOptimize()

      setMessage("Step 2/3: Filtering candidates...")
      setProgress(40)
      const filterData = await runFilter(optimizeData)

      setMessage("Step 3/3: Clustering into panel...")
      setProgress(70)
      await runCluster(filterData)

      setQuickStatus("success")
      setProgress(100)
      setMessage("Pipeline complete!")

      toast({
        title: "Success",
        description: "Quick pipeline completed successfully",
      })
    } catch (error) {
      setQuickStatus("error")
      setMessage("Pipeline failed")
      toast({
        title: "Error",
        description: "Pipeline execution failed",
        variant: "destructive",
      })
    }
  }

  const StatusIcon = ({ status }: { status: JobStatus }) => {
    if (status === "success") return <CheckCircle2 className="h-5 w-5 text-green-500" />
    if (status === "running") return <Loader2 className="h-5 w-5 animate-spin text-blue-500" />
    return <Circle className="h-5 w-5 text-muted-foreground" />
  }

  return (
    <div className="space-y-6">
      <Card className="border-border bg-background">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Zap className="h-5 w-5 text-blue-500" />
            Pipeline Overview
          </CardTitle>
          <CardDescription>Track your CDR3 generation workflow</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
            <div className="flex flex-1 items-center gap-3">
              <StatusIcon status={optimizeStatus} />
              <div className="flex-1">
                <p className="font-medium">Generate</p>
                {stats.generated > 0 && (
                  <Badge variant="secondary" className="mt-1">
                    {stats.generated} candidates
                  </Badge>
                )}
              </div>
            </div>

            <ArrowRight className="hidden h-5 w-5 text-muted-foreground md:block" />

            <div className="flex flex-1 items-center gap-3">
              <StatusIcon status={filterStatus} />
              <div className="flex-1">
                <p className="font-medium">Filter</p>
                {stats.filtered > 0 && (
                  <Badge variant="secondary" className="mt-1">
                    {stats.filtered} top
                  </Badge>
                )}
              </div>
            </div>

            <ArrowRight className="hidden h-5 w-5 text-muted-foreground md:block" />

            <div className="flex flex-1 items-center gap-3">
              <StatusIcon status={clusterStatus} />
              <div className="flex-1">
                <p className="font-medium">Cluster</p>
                {stats.clustered > 0 && (
                  <Badge variant="secondary" className="mt-1">
                    {stats.clustered} clusters
                  </Badge>
                )}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="grid gap-4 md:gap-6">
        <div className="grid gap-4 md:gap-6 lg:grid-cols-3">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-lg md:text-xl">
                <Play className="h-5 w-5 text-blue-500" />
                Step 1: Generate
              </CardTitle>
              <CardDescription>Run beam search optimization to generate CDR3 candidates</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <Button onClick={runOptimize} disabled={optimizeStatus === "running"} className="w-full h-12" size="lg">
                {optimizeStatus === "running" ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Generating...
                  </>
                ) : (
                  <>
                    <Play className="mr-2 h-4 w-4" />
                    Generate Candidates
                  </>
                )}
              </Button>
              <div className="rounded-lg border border-border bg-muted/50 p-3">
                <p className="text-xs text-muted-foreground">
                  Uses beam search to explore sequence space and generate diverse CDR3 candidates optimized for target
                  antigen binding.
                </p>
              </div>
              {optimizeStatus === "success" && (
                <Alert className="border-green-500/50 bg-green-950/20">
                  <CheckCircle2 className="h-4 w-4 text-green-500" />
                  <AlertDescription>Candidates generated successfully</AlertDescription>
                </Alert>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-lg md:text-xl">
                <Filter className="h-5 w-5 text-blue-500" />
                Step 2: Filter
              </CardTitle>
              <CardDescription>Filter and rank candidates by score</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <Button
                onClick={() => runFilter()}
                disabled={filterStatus === "running"}
                className="w-full h-12"
                size="lg"
                variant="secondary"
              >
                {filterStatus === "running" ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Filtering...
                  </>
                ) : (
                  <>
                    <Filter className="mr-2 h-4 w-4" />
                    Filter Candidates
                  </>
                )}
              </Button>
              <div className="rounded-lg border border-border bg-muted/50 p-3">
                <p className="text-xs text-muted-foreground">
                  Ranks candidates using the trained scoring model and selects the top performers based on predicted
                  binding affinity.
                </p>
              </div>
              {filterStatus === "success" && (
                <Alert className="border-green-500/50 bg-green-950/20">
                  <CheckCircle2 className="h-4 w-4 text-green-500" />
                  <AlertDescription>Candidates filtered successfully</AlertDescription>
                </Alert>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-lg md:text-xl">
                <Grid3x3 className="h-5 w-5 text-blue-500" />
                Step 3: Cluster
              </CardTitle>
              <CardDescription>Create diverse panel using k-means clustering</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <Button
                onClick={() => runCluster()}
                disabled={clusterStatus === "running"}
                className="w-full h-12"
                size="lg"
                variant="secondary"
              >
                {clusterStatus === "running" ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Clustering...
                  </>
                ) : (
                  <>
                    <Grid3x3 className="mr-2 h-4 w-4" />
                    Create Panel
                  </>
                )}
              </Button>
              <div className="rounded-lg border border-border bg-muted/50 p-3">
                <p className="text-xs text-muted-foreground">
                  Groups similar sequences using k-means clustering to create a diverse panel representing different
                  binding modes.
                </p>
              </div>
              {clusterStatus === "success" && (
                <Alert className="border-green-500/50 bg-green-950/20">
                  <CheckCircle2 className="h-4 w-4 text-green-500" />
                  <AlertDescription>Panel created successfully</AlertDescription>
                </Alert>
              )}
            </CardContent>
          </Card>
        </div>

        <Card className="border-border bg-background">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-lg md:text-xl">
              <Zap className="h-5 w-5 text-blue-500" />
              Quick Pipeline
            </CardTitle>
            <CardDescription>Run all three steps automatically in sequence</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <Button onClick={runQuickPipeline} disabled={quickStatus === "running"} className="w-full h-12" size="lg">
              {quickStatus === "running" ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Running Pipeline...
                </>
              ) : (
                <>
                  <Zap className="mr-2 h-4 w-4" />
                  Run Quick Pipeline
                </>
              )}
            </Button>

            {quickStatus === "running" && (
              <div className="space-y-2">
                <Progress value={progress} className="h-2" />
                <p className="text-sm text-muted-foreground">{message}</p>
              </div>
            )}

            {quickStatus === "success" && (
              <Alert className="border-green-500/50 bg-green-950/20">
                <CheckCircle2 className="h-4 w-4 text-green-500" />
                <AlertDescription>{message}</AlertDescription>
              </Alert>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
