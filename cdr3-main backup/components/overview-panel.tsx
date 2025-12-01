"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Settings, Zap, FileText, GraduationCap, BarChart3, ArrowRight, Dna } from "lucide-react"

interface OverviewPanelProps {
  onNavigate: (tab: string) => void
}

export function OverviewPanel({ onNavigate }: OverviewPanelProps) {
  return (
    <div className="space-y-6">
      {/* Hero Section */}
      <Card className="border-primary/20 bg-gradient-to-br from-primary/5 via-background to-background">
        <CardHeader className="text-center space-y-4 pb-8">
          <div className="flex justify-center">
            <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-primary/10 ring-4 ring-primary/20">
              <Dna className="h-8 w-8 text-primary" />
            </div>
          </div>
          <div>
            <CardTitle className="text-3xl md:text-4xl font-bold mb-2">Welcome to ABYSS</CardTitle>
            <CardDescription className="text-base md:text-lg max-w-2xl mx-auto">
              Antibody Binding Yield through Sequence Searching
            </CardDescription>
          </div>
        </CardHeader>
      </Card>

      {/* What is CDR3 */}
      <Card className="border-primary/20">
        <CardHeader>
          <CardTitle className="text-2xl">What is CDR3?</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4 text-muted-foreground">
          <p>
            The <strong className="text-foreground">Complementarity-Determining Region 3 (CDR3)</strong> is the most
            variable region of an antibody and the primary determinant of antigen specificity. Located at the center of
            the antigen-binding site, CDR3 directly contacts the target antigen and determines binding affinity.
          </p>
          <p>
            CDR3 sequences are highly diverse due to V(D)J recombination and somatic hypermutation, making them ideal
            targets for rational antibody design. However, this diversity also makes it challenging to predict which
            CDR3 sequences will bind to a specific antigen.
          </p>
        </CardContent>
      </Card>

      {/* The Challenge */}
      <Card className="border-primary/20">
        <CardHeader>
          <CardTitle className="text-2xl">The Challenge</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4 text-muted-foreground">
          <p>
            Traditional antibody discovery relies on immunization, hybridoma technology, or phage display—processes that
            are time-consuming, expensive, and often yield suboptimal candidates. Computational approaches can
            accelerate discovery, but predicting antigen-specific CDR3 sequences remains difficult due to:
          </p>
          <ul className="list-disc list-inside space-y-2 ml-4">
            <li>
              The vast sequence space (20<sup>n</sup> possible sequences for length n)
            </li>
            <li>Complex structure-function relationships</li>
            <li>Limited training data for machine learning models</li>
            <li>The need for both high affinity and developability</li>
          </ul>
        </CardContent>
      </Card>

      {/* Our Solution */}
      <Card className="border-primary/20">
        <CardHeader>
          <CardTitle className="text-2xl">Our Solution</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4 text-muted-foreground">
          <p>
            ABYSS uses a <strong className="text-foreground">machine learning-guided optimization pipeline</strong> to
            generate antigen-specific CDR3 sequences de novo. The system combines:
          </p>
          <div className="grid gap-4 md:grid-cols-2 mt-4">
            <div className="space-y-2">
              <h4 className="font-semibold text-foreground">ESM-2 Protein Language Model</h4>
              <p className="text-sm">
                Leverages evolutionary information from millions of protein sequences to generate biologically plausible
                embeddings of antigen-CDR3 pairs.
              </p>
            </div>
            <div className="space-y-2">
              <h4 className="font-semibold text-foreground">Iterative Beam Search</h4>
              <p className="text-sm">
                Explores sequence space efficiently by maintaining top candidates at each step and applying targeted
                mutations guided by a learned scoring function.
              </p>
            </div>
            <div className="space-y-2">
              <h4 className="font-semibold text-foreground">Scoring Model</h4>
              <p className="text-sm">
                A neural network trained on antigen-CDR3 binding pairs that predicts binding likelihood from ESM-2
                embeddings, enabling rapid candidate evaluation.
              </p>
            </div>
            <div className="space-y-2">
              <h4 className="font-semibold text-foreground">Diversity Clustering</h4>
              <p className="text-sm">
                Groups similar candidates and selects diverse representatives to maximize coverage of promising sequence
                space regions.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Workflow Steps */}
      <Card>
        <CardHeader>
          <CardTitle className="text-2xl">Getting Started</CardTitle>
          <CardDescription>Follow these steps to design your CDR3 sequences</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {/* Step 1 */}
            <Card className="border-muted">
              <CardHeader>
                <div className="flex items-start justify-between mb-2">
                  <Badge variant="outline" className="bg-primary/10">
                    Step 1
                  </Badge>
                  <Settings className="h-5 w-5 text-muted-foreground" />
                </div>
                <CardTitle className="text-lg">Configure</CardTitle>
                <CardDescription>Set up your parameters and model settings</CardDescription>
              </CardHeader>
              <CardContent>
                <Button onClick={() => onNavigate("config")} variant="outline" className="w-full">
                  Go to Config
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              </CardContent>
            </Card>

            {/* Step 2 */}
            <Card className="border-muted">
              <CardHeader>
                <div className="flex items-start justify-between mb-2">
                  <Badge variant="outline" className="bg-primary/10">
                    Step 2
                  </Badge>
                  <Zap className="h-5 w-5 text-muted-foreground" />
                </div>
                <CardTitle className="text-lg">Optimize</CardTitle>
                <CardDescription>Generate and filter CDR3 candidates</CardDescription>
              </CardHeader>
              <CardContent>
                <Button onClick={() => onNavigate("optimize")} variant="outline" className="w-full">
                  Go to Optimize
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              </CardContent>
            </Card>

            {/* Step 3 */}
            <Card className="border-muted">
              <CardHeader>
                <div className="flex items-start justify-between mb-2">
                  <Badge variant="outline" className="bg-primary/10">
                    Step 3
                  </Badge>
                  <FileText className="h-5 w-5 text-muted-foreground" />
                </div>
                <CardTitle className="text-lg">Results</CardTitle>
                <CardDescription>View and download your sequences</CardDescription>
              </CardHeader>
              <CardContent>
                <Button onClick={() => onNavigate("results")} variant="outline" className="w-full">
                  Go to Results
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              </CardContent>
            </Card>
          </div>

          {/* Advanced Options */}
          <div className="pt-4 border-t">
            <h3 className="text-sm font-semibold mb-3 text-muted-foreground uppercase tracking-wide">
              Advanced Options
            </h3>
            <div className="grid gap-3 md:grid-cols-2">
              <Button onClick={() => onNavigate("train")} variant="outline" className="justify-start h-auto py-3">
                <GraduationCap className="mr-3 h-5 w-5 text-primary" />
                <div className="text-left">
                  <div className="font-semibold">Train Model</div>
                  <div className="text-xs text-muted-foreground">Train on custom antigen-CDR3 pairs</div>
                </div>
              </Button>

              <Button onClick={() => onNavigate("evaluate")} variant="outline" className="justify-start h-auto py-3">
                <BarChart3 className="mr-3 h-5 w-5 text-primary" />
                <div className="text-left">
                  <div className="font-semibold">Evaluate</div>
                  <div className="text-xs text-muted-foreground">Assess model performance metrics</div>
                </div>
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Quick Stats / Info */}
      <div className="grid gap-4 md:grid-cols-3">
        <Card className="bg-gradient-to-br from-primary/10 to-background border-primary/20">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium text-muted-foreground">Pipeline</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">3 Steps</div>
            <p className="text-xs text-muted-foreground mt-1">Generate → Filter → Cluster</p>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-primary/10 to-background border-primary/20">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium text-muted-foreground">Model Type</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">ESM-2</div>
            <p className="text-xs text-muted-foreground mt-1">Protein language model</p>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-primary/10 to-background border-primary/20">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium text-muted-foreground">Output</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">CSV + FASTA</div>
            <p className="text-xs text-muted-foreground mt-1">Downloadable results</p>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
