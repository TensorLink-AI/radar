// Dashboard frontend glue: renders the loss curve on the experiment detail
// page and the Pareto scatter on /dashboard/pareto. Intentionally minimal —
// no build step, just Chart.js off the CDN.

(function () {
    "use strict";

    // ── Loss curve on experiment detail ──
    const canvas = document.getElementById("loss-curve");
    if (canvas) {
        const idx = canvas.dataset.index;
        fetch(`/dashboard/api/loss_curve/${idx}.json`)
            .then((r) => r.json())
            .then((data) => {
                const pts = data.points || [];
                if (!pts.length) {
                    canvas.replaceWith(
                        Object.assign(document.createElement("p"), {
                            className: "muted",
                            textContent: "No loss curve recorded.",
                        }),
                    );
                    return;
                }
                new Chart(canvas, {
                    type: "line",
                    data: {
                        labels: pts.map((_, i) => i),
                        datasets: [{
                            data: pts,
                            borderColor: "#6ea8fe",
                            backgroundColor: "rgba(110,168,254,0.15)",
                            tension: 0.15, pointRadius: 0, borderWidth: 2,
                        }],
                    },
                    options: {
                        plugins: { legend: { display: false } },
                        scales: {
                            x: { ticks: { color: "#8a8f99" } },
                            y: { ticks: { color: "#8a8f99" } },
                        },
                    },
                });
            })
            .catch((err) => console.warn("loss curve fetch failed:", err));
    }

    // ── Pareto scatter ──
    const scatter = document.getElementById("pareto-scatter");
    if (scatter) {
        const task = scatter.dataset.task || "";
        const qs = task ? `?task=${encodeURIComponent(task)}` : "";
        fetch(`/dashboard/api/pareto.json${qs}`)
            .then((r) => r.json())
            .then((data) => {
                const pts = data.points || [];
                const frontier = pts.filter((p) => p.on_frontier);
                const dominated = pts.filter((p) => !p.on_frontier);
                new Chart(scatter, {
                    type: "scatter",
                    data: {
                        datasets: [
                            {
                                label: `frontier (${frontier.length})`,
                                data: frontier.map((p) => ({x: p.flops, y: p.metric, id: p.id, name: p.name})),
                                backgroundColor: "#6ea8fe",
                                pointRadius: 5,
                            },
                            {
                                label: `dominated (${dominated.length})`,
                                data: dominated.map((p) => ({x: p.flops, y: p.metric, id: p.id, name: p.name})),
                                backgroundColor: "rgba(138,143,153,0.5)",
                                pointRadius: 3,
                            },
                        ],
                    },
                    options: {
                        scales: {
                            x: {
                                type: "logarithmic",
                                title: { display: true, text: "FLOPs-equivalent size", color: "#8a8f99" },
                                ticks: { color: "#8a8f99" },
                            },
                            y: {
                                title: { display: true, text: "metric (lower is better)", color: "#8a8f99" },
                                ticks: { color: "#8a8f99" },
                            },
                        },
                        plugins: {
                            legend: { labels: { color: "#e6e6e6" } },
                            tooltip: {
                                callbacks: {
                                    label: (ctx) => {
                                        const p = ctx.raw;
                                        return `#${p.id} ${p.name || ""} — ${p.y.toFixed(4)} @ ${p.x}`;
                                    },
                                },
                            },
                        },
                        onClick: (_evt, els) => {
                            if (!els.length) return;
                            const pt = els[0].element.$context.raw;
                            if (pt && pt.id !== undefined) {
                                window.location = `/dashboard/experiments/${pt.id}`;
                            }
                        },
                    },
                });
            })
            .catch((err) => console.warn("pareto fetch failed:", err));
    }

    // ── Highlight HTMX-swapped diffs with Prism ──
    document.body.addEventListener("htmx:afterSwap", () => {
        if (window.Prism) window.Prism.highlightAll();
    });
})();
