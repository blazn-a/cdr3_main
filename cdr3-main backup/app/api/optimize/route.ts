import { NextResponse } from "next/server"

function generateMockCandidates(antigen: string, beam: number) {
  const aminoAcids = "ACDEFGHIKLMNPQRSTVWY"
  const candidates = []

  for (let i = 0; i < beam; i++) {
    let cdr3 = ""
    const length = 12 + Math.floor(Math.random() * 6) // 12-17 AA
    for (let j = 0; j < length; j++) {
      cdr3 += aminoAcids[Math.floor(Math.random() * aminoAcids.length)]
    }
    candidates.push({
      cdr3,
      score: 0.5 + Math.random() * 0.5, // 0.5-1.0
      step: Math.floor(Math.random() * 100),
    })
  }

  return candidates
}

export async function POST(request: Request) {
  try {
    const body = await request.json()
    const config = body.config || {
      antigen: "EXAMPLE",
      seed_cdr3: "CARDGYW",
      beam: 100,
      steps: 50,
      k_mut: 3,
    }

    if (!config.antigen) {
      return NextResponse.json({ error: "No antigen sequence configured" }, { status: 400 })
    }

    const candidates = generateMockCandidates(config.antigen, config.beam || 100)

    const timestamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, -5)

    return NextResponse.json({
      success: true,
      output: `opt_${timestamp}.csv`,
      count: candidates.length,
      demo: true,
      data: {
        timestamp,
        antigen: config.antigen,
        candidates,
        count: candidates.length,
      },
    })
  } catch (error) {
    console.error("[v0] Optimize error:", error)
    return NextResponse.json(
      {
        error: error instanceof Error ? error.message : "Optimization failed",
      },
      { status: 500 },
    )
  }
}
