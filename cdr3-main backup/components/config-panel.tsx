"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Button } from "@/components/ui/button"
import { Switch } from "@/components/ui/switch"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { useToast } from "@/hooks/use-toast"
import { Loader2, Save } from "lucide-react"

interface Config {
  antigen: string
  seed_cdr3: string
  esm: string
  checkpoint: string
  steps: number
  beam: number
  k_mut: number
  keep_top: number
  embed_batch: number
  clusters: number
  out_folder: string
  pairs_csv: string
  epochs: number
  train_batch: number
  threads: number
  use_gpu: boolean
  hard_neg: boolean
  hn_factor: number
  pos_mult: number
}

const DEFAULTS: Config = {
  antigen: "",
  seed_cdr3: "CARDRSTGYVYFDYW",
  esm: "t12_35M",
  checkpoint: "score_model.pt",
  steps: 200,
  beam: 50,
  k_mut: 1,
  keep_top: 50,
  embed_batch: 64,
  clusters: 12,
  out_folder: "runs",
  pairs_csv: "data/antigen_cdr3_pairs.csv",
  epochs: 30,
  train_batch: 128,
  threads: 2,
  use_gpu: false,
  hard_neg: true,
  hn_factor: 1,
  pos_mult: 1,
}

export function ConfigPanel() {
  const { toast } = useToast()
  const [config, setConfig] = useState<Config | null>(null)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    loadConfig()
  }, [])

  const loadConfig = () => {
    try {
      const stored = sessionStorage.getItem("abyss_config")
      if (stored) {
        setConfig(JSON.parse(stored))
      } else {
        setConfig(DEFAULTS)
      }
    } catch (error) {
      console.error("[v0] Failed to load config from sessionStorage:", error)
      setConfig(DEFAULTS)
      toast({
        title: "Error",
        description: "Failed to load configuration, using defaults",
        variant: "destructive",
      })
    } finally {
      setLoading(false)
    }
  }

  const saveConfig = () => {
    if (!config) return

    setSaving(true)
    try {
      sessionStorage.setItem("abyss_config", JSON.stringify(config))
      toast({
        title: "Success",
        description: "Configuration saved successfully",
      })
    } catch (error) {
      console.error("[v0] Failed to save config to sessionStorage:", error)
      toast({
        title: "Error",
        description: "Failed to save configuration",
        variant: "destructive",
      })
    } finally {
      setSaving(false)
    }
  }

  const validateAminoAcids = (value: string): string => {
    return value.toUpperCase().replace(/[^ACDEFGHIKLMNPQRSTVWY]/g, "")
  }

  if (loading || !config) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center py-12">
          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="grid gap-4 md:gap-6 lg:grid-cols-2">
      {/* Core Settings */}
      <Card>
        <CardHeader>
          <CardTitle>Core Settings</CardTitle>
          <CardDescription>Primary configuration for CDR3 optimization</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="antigen">Antigen Sequence</Label>
            <Textarea
              id="antigen"
              placeholder="Paste antigen sequence (amino acids only)"
              value={config.antigen}
              onChange={(e) => setConfig({ ...config, antigen: validateAminoAcids(e.target.value) })}
              className="font-mono text-sm min-h-[80px]"
              rows={3}
            />
            <p className="text-xs text-muted-foreground">Length: {config.antigen.length} AA</p>
          </div>

          <div className="space-y-2">
            <Label htmlFor="seed_cdr3">Seed CDR3</Label>
            <Input
              id="seed_cdr3"
              value={config.seed_cdr3}
              onChange={(e) => setConfig({ ...config, seed_cdr3: validateAminoAcids(e.target.value) })}
              className="font-mono"
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="esm">ESM Backbone</Label>
            <Select value={config.esm} onValueChange={(value) => setConfig({ ...config, esm: value })}>
              <SelectTrigger id="esm">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="t6_8M">t6_8M (Faster)</SelectItem>
                <SelectItem value="t12_35M">t12_35M (Better)</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="checkpoint">Checkpoint Path</Label>
            <Input
              id="checkpoint"
              value={config.checkpoint}
              onChange={(e) => setConfig({ ...config, checkpoint: e.target.value })}
            />
          </div>
        </CardContent>
      </Card>

      {/* Generation Parameters */}
      <Card>
        <CardHeader>
          <CardTitle>Generation Parameters</CardTitle>
          <CardDescription>Control the optimization process</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-3 md:gap-4">
            <div className="space-y-2">
              <Label htmlFor="steps" className="text-sm">
                Steps
              </Label>
              <Input
                id="steps"
                type="number"
                value={config.steps}
                onChange={(e) => setConfig({ ...config, steps: Number.parseInt(e.target.value) || 0 })}
                className="h-10"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="beam" className="text-sm">
                Beam Width
              </Label>
              <Input
                id="beam"
                type="number"
                value={config.beam}
                onChange={(e) => setConfig({ ...config, beam: Number.parseInt(e.target.value) || 0 })}
                className="h-10"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="k_mut" className="text-sm">
                Mutations/Step
              </Label>
              <Input
                id="k_mut"
                type="number"
                value={config.k_mut}
                onChange={(e) => setConfig({ ...config, k_mut: Number.parseInt(e.target.value) || 0 })}
                className="h-10"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="keep_top" className="text-sm">
                Keep Top
              </Label>
              <Input
                id="keep_top"
                type="number"
                value={config.keep_top}
                onChange={(e) => setConfig({ ...config, keep_top: Number.parseInt(e.target.value) || 0 })}
                className="h-10"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="clusters" className="text-sm">
                Clusters (k)
              </Label>
              <Input
                id="clusters"
                type="number"
                value={config.clusters}
                onChange={(e) => setConfig({ ...config, clusters: Number.parseInt(e.target.value) || 0 })}
                className="h-10"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="embed_batch" className="text-sm">
                Embed Batch
              </Label>
              <Input
                id="embed_batch"
                type="number"
                value={config.embed_batch}
                onChange={(e) => setConfig({ ...config, embed_batch: Number.parseInt(e.target.value) || 0 })}
                className="h-10"
              />
            </div>
          </div>

          <div className="space-y-2">
            <Label htmlFor="out_folder">Output Folder</Label>
            <Input
              id="out_folder"
              value={config.out_folder}
              onChange={(e) => setConfig({ ...config, out_folder: e.target.value })}
            />
          </div>
        </CardContent>
      </Card>

      {/* Training Settings */}
      <Card className="lg:col-span-2">
        <CardHeader>
          <CardTitle>Training Settings</CardTitle>
          <CardDescription>Configure model training parameters</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-4 lg:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="pairs_csv">Pairs CSV Path</Label>
              <Input
                id="pairs_csv"
                value={config.pairs_csv}
                onChange={(e) => setConfig({ ...config, pairs_csv: e.target.value })}
              />
            </div>

            <div className="grid grid-cols-2 gap-3 md:gap-4">
              <div className="space-y-2">
                <Label htmlFor="epochs" className="text-sm">
                  Epochs
                </Label>
                <Input
                  id="epochs"
                  type="number"
                  value={config.epochs}
                  onChange={(e) => setConfig({ ...config, epochs: Number.parseInt(e.target.value) || 0 })}
                  className="h-10"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="train_batch" className="text-sm">
                  Batch Size
                </Label>
                <Input
                  id="train_batch"
                  type="number"
                  value={config.train_batch}
                  onChange={(e) => setConfig({ ...config, train_batch: Number.parseInt(e.target.value) || 0 })}
                  className="h-10"
                />
              </div>
            </div>
          </div>

          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            <div className="space-y-2">
              <Label htmlFor="threads" className="text-sm">
                Threads
              </Label>
              <Input
                id="threads"
                type="number"
                value={config.threads}
                onChange={(e) => setConfig({ ...config, threads: Number.parseInt(e.target.value) || 0 })}
                className="h-10"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="hn_factor" className="text-sm">
                Hard-Neg Factor
              </Label>
              <Input
                id="hn_factor"
                type="number"
                value={config.hn_factor}
                onChange={(e) => setConfig({ ...config, hn_factor: Number.parseInt(e.target.value) || 0 })}
                className="h-10"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="pos_mult" className="text-sm">
                Pos Oversample
              </Label>
              <Input
                id="pos_mult"
                type="number"
                value={config.pos_mult}
                onChange={(e) => setConfig({ ...config, pos_mult: Number.parseInt(e.target.value) || 1 })}
                className="h-10"
              />
            </div>

            <div className="space-y-2">
              <Label className="text-sm">GPU / Hard-Neg</Label>
              <div className="flex gap-2">
                <div className="flex items-center gap-2 rounded-lg border border-border px-3 py-2">
                  <Label htmlFor="use_gpu" className="text-xs cursor-pointer">
                    GPU
                  </Label>
                  <Switch
                    id="use_gpu"
                    checked={config.use_gpu}
                    onCheckedChange={(checked) => setConfig({ ...config, use_gpu: checked })}
                  />
                </div>
                <div className="flex items-center gap-2 rounded-lg border border-border px-3 py-2">
                  <Label htmlFor="hard_neg" className="text-xs cursor-pointer">
                    Hard-Neg
                  </Label>
                  <Switch
                    id="hard_neg"
                    checked={config.hard_neg}
                    onCheckedChange={(checked) => setConfig({ ...config, hard_neg: checked })}
                  />
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Save Button */}
      <div className="lg:col-span-2">
        <Button onClick={saveConfig} disabled={saving} className="w-full h-12 md:h-auto" size="lg">
          {saving ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Saving...
            </>
          ) : (
            <>
              <Save className="mr-2 h-4 w-4" />
              Save Configuration
            </>
          )}
        </Button>
      </div>
    </div>
  )
}
