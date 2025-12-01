import { NextResponse } from "next/server"
import { promises as fs } from "fs"
import path from "path"

export async function GET(request: Request, { params }: { params: { type: string } }) {
  try {
    const configData = await fs.readFile(path.join(process.cwd(), "ui_state.json"), "utf-8")
    const config = JSON.parse(configData)

    let filePath: string | null = null

    switch (params.type) {
      case "raw":
        filePath = config.last_raw
        break
      case "filtered":
        filePath = config.last_filtered
        break
      case "panel":
        filePath = config.last_panel
        break
      case "fasta":
        if (config.last_panel) {
          filePath = config.last_panel.replace(".csv", ".fasta")
        }
        break
    }

    if (!filePath) {
      return NextResponse.json({ error: "File not found" }, { status: 404 })
    }

    const fullPath = path.join(process.cwd(), filePath)
    const data = await fs.readFile(fullPath)

    return new NextResponse(data, {
      headers: {
        "Content-Type": params.type === "fasta" ? "text/plain" : "text/csv",
        "Content-Disposition": `attachment; filename="${path.basename(filePath)}"`,
      },
    })
  } catch (error) {
    return NextResponse.json({ error: "File not found" }, { status: 404 })
  }
}
