import { NextResponse } from "next/server"
import { exec } from "child_process"
import { promisify } from "util"
import { promises as fs } from "fs"
import path from "path"

const execAsync = promisify(exec)

export async function POST() {
  try {
    const configData = await fs.readFile(path.join(process.cwd(), "ui_state.json"), "utf-8")
    const config = JSON.parse(configData)

    // Check if pairs CSV exists
    const pairsPath = path.join(process.cwd(), config.pairs_csv)
    try {
      await fs.access(pairsPath)
    } catch {
      return NextResponse.json(
        { error: "Pairs CSV not found. Please add antigen_cdr3_pairs.csv to data folder." },
        { status: 400 },
      )
    }

    // Step 1: Embed pairs
    const embedArgs = [
      `--csv "${config.pairs_csv}"`,
      '--emb_out "data/pair_embeddings.npy"',
      '--lab_out "data/labels.npy"',
      `--batch ${config.embed_batch}`,
      `--threads ${config.threads}`,
      `--model "${config.esm}"`,
      config.use_gpu ? "" : "--force_cpu",
    ]
      .filter(Boolean)
      .join(" ")

    await execAsync(`python esm_embeddings.py ${embedArgs}`)

    // Step 2: Train model
    await execAsync(`python score_model.py --epochs ${config.epochs} --batch ${config.train_batch}`)

    // Step 3: Get AUC
    let auc = null
    try {
      const { stdout } = await execAsync("python eval_auc.py")
      const match = stdout.match(/AUC[:\s]+([0-9.]+)/)
      if (match) auc = Number.parseFloat(match[1])
    } catch {
      console.log("[v0] Could not get AUC")
    }

    // Update checkpoint in config
    config.checkpoint = "score_model.pt"
    await fs.writeFile(path.join(process.cwd(), "ui_state.json"), JSON.stringify(config, null, 2))

    return NextResponse.json({ success: true, auc })
  } catch (error) {
    console.error("[v0] Train error:", error)
    return NextResponse.json({ error: error instanceof Error ? error.message : "Training failed" }, { status: 500 })
  }
}
