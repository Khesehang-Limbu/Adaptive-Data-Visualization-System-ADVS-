document.addEventListener("DOMContentLoaded", function () {
    setTimeout(() => {
        const alert = document.querySelector(".alert");
        if (alert) {
            alert.style.display = "none";
        }
    }, 3000);
})

function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    sidebar.classList.toggle('hidden');
}

function renderPlotlyCharts(charts) {
    if (charts === "None") {
        return;
    }
    console.log(charts)
    charts.forEach((chart, idx) => {
        const el = document.getElementById(`chart-ctx-${chart.context_id}-${idx}`);
        if (!el) return;

        Plotly.newPlot(el, chart.plotly_data, chart.plotly_layout, {responsive: true});
    });
}

window.addEventListener('DOMContentLoaded', () => {
    if (window.initial_charts && window.initial_charts.length) {
        renderPlotlyCharts(window.initial_charts);
    }
});

document.addEventListener("htmx:afterSettle", function () {
    const charts = window.initial_charts || [];
    renderPlotlyCharts(charts);
});

document.addEventListener('charts:render', function (event) {
    const charts = (event.detail && event.detail.chart_Config) || window.initial_charts || [];
    renderPlotlyCharts(charts);
});

document.body.addEventListener("htmx:afterSwap", function (event) {
    console.log(event);
    if (event.target.matches("[id^='context-']")) {
        event.target.querySelectorAll("[id^='chart-ctx-']").forEach(el => {
            const chartData = JSON.parse(el.dataset.plotlyData);
            const chartLayout = JSON.parse(el.dataset.plotlyLayout);
            Plotly.newPlot(el.id, chartData, chartLayout);
        });
    }
});
