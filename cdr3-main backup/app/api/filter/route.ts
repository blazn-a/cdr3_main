import { NextResponse } from "next/server"

export async function POST(request: Request) {
  try {
    const body = await request.json()
    const rawData = body.data
    const config = body.config || { keep_top: 50 }

    if (!rawData || !rawData.candidates) {
      return NextResponse.json(
        {
          error: "No raw candidates found. Run optimization first.",
        },
        { status: 400 },
      )
    }

    const sorted = [...rawData.candidates].sort((a: any, b: any) => b.score - a.score)
    const filtered = sorted.slice(0, config.keep_top || 50)

    const resultData = {
      timestamp: rawData.timestamp,
      antigen: rawData.antigen,
      candidates: filtered,
      count: filtered.length,
    }

    return NextResponse.json({
      success: true,
      output: `opt_${rawData.timestamp}_filtered.csv`,
      count: filtered.length,
      demo: true,
      data: resultData,
    })
  } catch (error) {
    console.error("[v0] Filter error:", error)
    return NextResponse.json(
      {
        error: error instanceof Error ? error.message : "Filtering failed",
      },
      { status: 500 },
    )
  }
}
