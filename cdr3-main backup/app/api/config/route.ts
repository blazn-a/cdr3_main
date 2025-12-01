import { NextResponse } from "next/server"
import { promises as fs } from "fs"
import path from "path"

const STATE_FILE = path.join(process.cwd(), "ui_state.json")

const DEFAULTS = {
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
  last_raw: "",
  last_filtered: "",
  last_panel: "",
}

export async function GET() {
  try {
    const data = await fs.readFile(STATE_FILE, "utf-8")
    const config = { ...DEFAULTS, ...JSON.parse(data) }
    return NextResponse.json(config)
  } catch (error) {
    return NextResponse.json(DEFAULTS)
  }
}

export async function POST(request: Request) {
  try {
    const config = await request.json()
    await fs.writeFile(STATE_FILE, JSON.stringify(config, null, 2))
    return NextResponse.json({ success: true })
  } catch (error) {
    return NextResponse.json({ error: "Failed to save configuration" }, { status: 500 })
  }
}
