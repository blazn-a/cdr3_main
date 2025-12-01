import { NextResponse } from "next/server"

export async function GET() {
  try {
    const rawDataStr = typeof sessionStorage !== "undefined" ? sessionStorage.getItem("abyss_last_raw") : null
    const filteredDataStr = typeof sessionStorage !== "undefined" ? sessionStorage.getItem("abyss_last_filtered") : null
    const panelDataStr = typeof sessionStorage !== "undefined" ? sessionStorage.getItem("abyss_last_panel") : null

    const results: any = {
      raw: rawDataStr ? "Generated" : null,
      filtered: filteredDataStr ? "Filtered" : null,
      panel: panelDataStr ? "Panel Created" : null,
      candidates: [],
    }

    // Load panel candidates for preview
    if (panelDataStr) {
      try {
        const panelData = JSON.parse(panelDataStr)
        results.candidates = panelData.candidates.slice(0, 20).map((c: any) => ({
          cdr3: c.cdr3,
          score: c.score,
          cluster: c.cluster,
        }))
      } catch (error) {
        console.log("[v0] Could not load candidates preview:", error)
      }
    }

    return NextResponse.json(results)
  } catch (error) {
    return NextResponse.json({ error: "Failed to load results" }, { status: 500 })
  }
}
