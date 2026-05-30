from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path

import markdown

from crew_ai_analysis_crew.schemas import NerResult, SentimentResult, SimilarityResult


def _chart_colors(n: int) -> list[str]:
    palette = [
        "#4f46e5",
        "#06b6d4",
        "#10b981",
        "#f59e0b",
        "#ef4444",
        "#8b5cf6",
        "#ec4899",
        "#64748b",
    ]
    return [palette[i % len(palette)] for i in range(max(n, 1))]


def _entity_section_html(ner: NerResult) -> str:
    if not ner.entities_by_label:
        return "<p class='muted'>No named entities detected.</p>"

    parts: list[str] = [
        f"<p class='stat-inline'>Total entities: <strong>{ner.entity_count}</strong></p>"
    ]
    for label in sorted(ner.entities_by_label.keys()):
        names = ner.entities_by_label[label]
        chips = "".join(f"<span class='chip'>{html.escape(name)}</span>" for name in names)
        parts.append(
            f"<div class='entity-group'><h4>{html.escape(label)} "
            f"<span class='badge'>{len(names)}</span></h4>"
            f"<div class='chips'>{chips}</div></div>"
        )
    return "\n".join(parts)


def _similarity_section_html(similarity: SimilarityResult) -> str:
    if not similarity.matches:
        return f"<p class='muted'>{html.escape(similarity.message or 'No similar documents found.')}</p>"

    cards: list[str] = [
        f"<p class='muted'>{html.escape(similarity.message)}</p>",
    ]
    for match in similarity.matches:
        cards.append(
            f"<article class='match-card'>"
            f"<header><span class='match-name'>{html.escape(Path(match.path).name)}</span>"
            f"<span class='match-score'>{match.score:.1%}</span></header>"
            f"<p>{html.escape(match.snippet)}</p>"
            f"</article>"
        )
    return "\n".join(cards)


def build_html_report(
    document_title: str,
    ner: NerResult,
    sentiment: SentimentResult,
    similarity: SimilarityResult,
    executive_narrative: str,
) -> str:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    narrative_html = markdown.markdown(
        executive_narrative,
        extensions=["extra", "nl2br", "sane_lists"],
    )

    ner_labels = sorted(ner.entities_by_label.keys())
    ner_counts = [len(ner.entities_by_label[label]) for label in ner_labels]
    sent_labels = list(sentiment.scores.keys())
    sent_values = [round(v * 100, 2) for v in sentiment.scores.values()]
    sim_labels = [Path(m.path).name for m in similarity.matches]
    sim_values = [round(m.score * 100, 2) for m in similarity.matches]
    colors = _chart_colors(max(len(ner_labels), len(sent_labels), len(sim_labels), 3))

    sentiment_badge = {
        "positive": "badge-positive",
        "negative": "badge-negative",
        "neutral": "badge-neutral",
    }.get(sentiment.label.lower(), "badge-neutral")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Document Intelligence — {html.escape(document_title)}</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <style>
    :root {{
      --bg: #0f172a; --surface: #1e293b; --surface-2: #334155;
      --text: #f1f5f9; --muted: #94a3b8; --accent: #818cf8;
      --positive: #34d399; --negative: #f87171; --neutral: #fbbf24;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: "Segoe UI", system-ui, sans-serif;
      background: linear-gradient(160deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
      color: var(--text); line-height: 1.6; padding: 2rem 1.25rem 3rem;
    }}
    .container {{ max-width: 1100px; margin: 0 auto; }}
    .hero, .card, .narrative {{
      background: var(--surface); border: 1px solid var(--surface-2);
      border-radius: 12px; padding: 1.5rem; margin-bottom: 1.25rem;
      box-shadow: 0 10px 40px rgba(0,0,0,.35);
    }}
    .hero h1 {{ font-size: 1.75rem; margin-bottom: 0.5rem; }}
    .meta {{ color: var(--muted); font-size: 0.9rem; }}
    .badge {{
      display: inline-block; padding: 0.35rem 0.85rem; border-radius: 999px;
      font-size: 0.85rem; font-weight: 600; text-transform: capitalize;
    }}
    .badge-positive {{ background: rgba(52,211,153,.2); color: var(--positive); }}
    .badge-negative {{ background: rgba(248,113,113,.2); color: var(--negative); }}
    .badge-neutral {{ background: rgba(251,191,36,.2); color: var(--neutral); }}
    .stat-row {{ display: flex; flex-wrap: wrap; gap: 0.75rem; margin-top: 1rem; }}
    .stat {{ background: var(--surface-2); padding: 0.5rem 1rem; border-radius: 8px; }}
    .grid {{
      display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      gap: 1.25rem; margin-bottom: 1.25rem;
    }}
    .card h2 {{
      font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.06em;
      color: var(--muted); margin-bottom: 1rem;
    }}
    .chart-wrap {{ height: 280px; position: relative; }}
    .narrative h2 {{ color: var(--accent); margin-bottom: 1rem; }}
    .narrative h3, .narrative h4 {{ margin: 1rem 0 0.5rem; }}
    .narrative ul, .narrative ol {{ margin: 0 0 1rem 1.25rem; }}
    .narrative p {{ margin-bottom: 0.75rem; }}
    .entity-group {{ margin-bottom: 1rem; }}
    .entity-group h4 {{ font-size: 0.9rem; margin-bottom: 0.5rem; }}
    .chips {{ display: flex; flex-wrap: wrap; gap: 0.4rem; }}
    .chip {{
      background: var(--surface-2); padding: 0.25rem 0.6rem;
      border-radius: 6px; font-size: 0.85rem;
    }}
    .match-card {{
      background: var(--surface-2); border-radius: 8px;
      padding: 1rem; margin-bottom: 0.75rem;
    }}
    .match-card header {{
      display: flex; justify-content: space-between; margin-bottom: 0.5rem;
      font-weight: 600;
    }}
    .match-score {{ color: var(--accent); }}
    .match-card p {{ color: var(--muted); font-size: 0.9rem; white-space: pre-wrap; }}
    .muted {{ color: var(--muted); }}
    footer {{ text-align: center; color: var(--muted); font-size: 0.8rem; margin-top: 2rem; }}
  </style>
</head>
<body>
  <div class="container">
    <header class="hero">
      <h1>Document Intelligence Report</h1>
      <p class="meta">Source: {html.escape(document_title)} · Generated {generated_at}</p>
      <div class="stat-row">
        <span class="badge {sentiment_badge}">{html.escape(sentiment.label)} sentiment</span>
        <span class="stat">Confidence: {sentiment.confidence:.1%}</span>
        <span class="stat">Entities: {ner.entity_count}</span>
        <span class="stat">Corpus size: {similarity.corpus_size}</span>
      </div>
      <p class="meta" style="margin-top:1rem">{html.escape(sentiment.insights)}</p>
    </header>

    <section class="grid">
      <div class="card"><h2>Sentiment distribution</h2>
        <div class="chart-wrap"><canvas id="sentimentChart"></canvas></div></div>
      <div class="card"><h2>Entities by type</h2>
        <div class="chart-wrap"><canvas id="nerChart"></canvas></div></div>
      <div class="card"><h2>Corpus similarity</h2>
        <div class="chart-wrap"><canvas id="similarityChart"></canvas></div></div>
    </section>

    <section class="narrative">
      <h2>Executive analysis</h2>
      {narrative_html}
    </section>

    <section class="card">
      <h2>All named entities (from spaCy)</h2>
      {_entity_section_html(ner)}
    </section>

    <section class="card">
      <h2>Similar corpus documents</h2>
      {_similarity_section_html(similarity)}
    </section>

    <footer>Multi-Task Document Intelligence Crew · CrewAI</footer>
  </div>
  <script>
    Chart.defaults.color = "#94a3b8";
    Chart.defaults.borderColor = "#334155";
    const colors = {json.dumps(colors)};

    new Chart(document.getElementById("sentimentChart"), {{
      type: "doughnut",
      data: {{
        labels: {json.dumps(sent_labels)},
        datasets: [{{ data: {json.dumps(sent_values)}, backgroundColor: colors.slice(0, {len(sent_labels)}), borderWidth: 0 }}],
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{ legend: {{ position: "bottom" }} }},
      }},
    }});

    new Chart(document.getElementById("nerChart"), {{
      type: "bar",
      data: {{
        labels: {json.dumps(ner_labels)},
        datasets: [{{ label: "Count", data: {json.dumps(ner_counts)}, backgroundColor: colors.slice(0, {len(ner_labels)}), borderRadius: 6 }}],
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{ y: {{ beginAtZero: true, ticks: {{ stepSize: 1 }} }} }},
      }},
    }});

    new Chart(document.getElementById("similarityChart"), {{
      type: "bar",
      data: {{
        labels: {json.dumps(sim_labels)},
        datasets: [{{ label: "Similarity %", data: {json.dumps(sim_values)}, backgroundColor: colors.slice(0, {len(sim_labels)}), borderRadius: 6 }}],
      }},
      options: {{
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{ x: {{ max: 100, ticks: {{ callback: (v) => v + "%" }} }} }},
      }},
    }});
  </script>
</body>
</html>
"""
