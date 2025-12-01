"use client"

import { useState } from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ConfigPanel } from "@/components/config-panel"
import { OptimizePanel } from "@/components/optimize-panel"
import { ResultsPanel } from "@/components/results-panel"
import { TrainPanel } from "@/components/train-panel"
import { EvaluatePanel } from "@/components/evaluate-panel"
import { OverviewPanel } from "@/components/overview-panel"
import { Home, Settings, Zap, FileText, GraduationCap, BarChart3 } from "lucide-react"

export default function Page() {
  const [activeTab, setActiveTab] = useState("overview")

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-50 safe-top">
        <div className="container mx-auto px-4 py-4 md:py-6">
          <div className="flex flex-col items-center gap-1 text-center">
            <h1 className="text-2xl md:text-3xl font-bold tracking-tight text-foreground">ABYSS</h1>
            <p className="text-xs md:text-sm text-muted-foreground leading-tight">
              Antibody Binding Yield through Sequence Searching
            </p>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="w-full px-3 py-4 md:px-6 md:py-8">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <div className="flex justify-center mb-4 md:mb-6">
            <TabsList className="h-auto flex-wrap justify-center gap-1 bg-muted/50 p-1.5 md:p-2">
              <TabsTrigger value="overview" className="flex items-center gap-2 px-3 py-2.5 md:px-4 md:py-2.5 min-h-11">
                <Home className="h-4 w-4" />
                <span className="hidden sm:inline">Overview</span>
              </TabsTrigger>
              <TabsTrigger value="config" className="flex items-center gap-2 px-3 py-2.5 md:px-4 md:py-2.5 min-h-11">
                <Settings className="h-4 w-4" />
                <span className="hidden sm:inline">Config</span>
              </TabsTrigger>
              <TabsTrigger value="optimize" className="flex items-center gap-2 px-3 py-2.5 md:px-4 md:py-2.5 min-h-11">
                <Zap className="h-4 w-4" />
                <span className="hidden sm:inline">Optimize</span>
              </TabsTrigger>
              <TabsTrigger value="results" className="flex items-center gap-2 px-3 py-2.5 md:px-4 md:py-2.5 min-h-11">
                <FileText className="h-4 w-4" />
                <span className="hidden sm:inline">Results</span>
              </TabsTrigger>
              <TabsTrigger value="train" className="flex items-center gap-2 px-3 py-2.5 md:px-4 md:py-2.5 min-h-11">
                <GraduationCap className="h-4 w-4" />
                <span className="hidden sm:inline">Train</span>
              </TabsTrigger>
              <TabsTrigger value="evaluate" className="flex items-center gap-2 px-3 py-2.5 md:px-4 md:py-2.5 min-h-11">
                <BarChart3 className="h-4 w-4" />
                <span className="hidden sm:inline">Evaluate</span>
              </TabsTrigger>
            </TabsList>
          </div>

          <div className="mt-4 md:mt-6">
            <TabsContent value="overview" className="mt-0">
              <OverviewPanel onNavigate={setActiveTab} />
            </TabsContent>

            <TabsContent value="config" className="mt-0">
              <ConfigPanel />
            </TabsContent>

            <TabsContent value="optimize" className="mt-0">
              <OptimizePanel />
            </TabsContent>

            <TabsContent value="results" className="mt-0">
              <ResultsPanel />
            </TabsContent>

            <TabsContent value="train" className="mt-0">
              <TrainPanel />
            </TabsContent>

            <TabsContent value="evaluate" className="mt-0">
              <EvaluatePanel />
            </TabsContent>
          </div>
        </Tabs>
      </main>
    </div>
  )
}
